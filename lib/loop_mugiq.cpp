#include <loop_mugiq.h>
#include <gamma.h>
#include <cublas_v2.h>
#include <hdf5.h>

template <typename Float, QudaFieldOrder fieldOrder>
Loop_Mugiq<Float, fieldOrder>::Loop_Mugiq(MugiqLoopParam *loopParams_,
                              Eigsolve_Mugiq *eigsolve_) :
  cPrm(nullptr),
  displace(nullptr),
  eigsolve(eigsolve_),
  refVec(nullptr),
  COMM_SPACE(MPI_COMM_NULL),
  space_rank(-1),
  space_size(-1),
  cRank(-1),
  tCoord(-1),
  COMM_TIME(MPI_COMM_NULL),
  time_rank(-1),
  time_size(-1),
  time_color(-1),
  IamTimeProcess(MUGIQ_BOOL_INVALID),
  commsAreSet(MUGIQ_BOOL_FALSE),
  dataPos_d(nullptr),
  dataPosMP_d(nullptr),
  dataMom_d(nullptr),
  dataPos(nullptr),
  dataMom_h(nullptr),
  dataMom(nullptr),
  dataMom_bcast(nullptr),
  nElemMomTot(0),
  nElemMomLoc(0),
  MomProjDone(MUGIQ_BOOL_FALSE),
  writeDataPos(loopParams_->writePosSpaceHDF5),
  writeDataMom(loopParams_->writeMomSpaceHDF5),
  momSpaceFilename(loopParams_->fname_mom_h5),
  posSpaceFilename(loopParams_->fname_pos_h5)
{
  printfQuda("\n*************************************************\n");
  printfQuda("%s: Creating Loop computation environment\n", __func__);

  if(eigsolve->useMGenv && eigsolve->computeCoarse) refVec = eigsolve->tmpCSF[0]; //mg_env->mg_solver->B[0];
  else refVec = eigsolve->eVecs[0];
  
  cPrm = new LoopComputeParam(loopParams_, refVec);
  setupComms();

  allocateDataMemory();
  copyGammaToConstMem();
  if(cPrm->doMomProj) createPhaseMatrix();
  
  printLoopComputeParams();

  if(cPrm->doNonLocal) displace = new Displace<Float,fieldOrder>(loopParams_,
								 refVec,
								 eigsolve->eVecs[0]->Precision());

  printfQuda("*************************************************\n\n");
}

template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::setupComms(){

  //-- Create space-communicator
  tCoord = comm_coord(3);
  cRank = comm_rank();
  MPI_Comm_split(MPI_COMM_WORLD, tCoord, cRank, &COMM_SPACE);
  MPI_Comm_rank(COMM_SPACE,&space_rank);
  MPI_Comm_size(COMM_SPACE,&space_size);
  
  //-- Create time communicator
  //-- Determine the "color" which distinguishes the "time" processes from the rest  time_color = comm_rank();   
  IamTimeProcess = MUGIQ_BOOL_FALSE;
  if( (comm_coord(0) == 0) &&
      (comm_coord(1) == 0) &&
      (comm_coord(2) == 0) ){
    time_color = (time_tag>comm_size()) ? time_tag : time_tag+comm_size();
    IamTimeProcess = MUGIQ_BOOL_TRUE;
  }
    
  MPI_Comm_split(MPI_COMM_WORLD, time_color, tCoord, &COMM_TIME);
  MPI_Comm_rank(COMM_TIME,&time_rank);
  MPI_Comm_size(COMM_TIME,&time_size);

  printfQuda("%s: MPI Communicators are set\n", __func__);

  commsAreSet = MUGIQ_BOOL_TRUE;
}


template <typename Float, QudaFieldOrder fieldOrder>
Loop_Mugiq<Float, fieldOrder>::~Loop_Mugiq(){

  freeDataMemory();

  if(cPrm->doNonLocal) delete displace;
  delete cPrm;
}


template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::allocateDataMemory(){

  nElemMomTotPerLoop = cPrm->nG * cPrm->Nmom * cPrm->totT;
  nElemMomLocPerLoop = cPrm->nG * cPrm->Nmom * cPrm->locT;
  nElemPosLocPerLoop = cPrm->nG * cPrm->locV4;
  
  nElemMomTot = nElemMomTotPerLoop * cPrm->nLoop;
  nElemMomLoc = nElemMomLocPerLoop * cPrm->nLoop;
  nElemPosLoc = nElemPosLocPerLoop * cPrm->nLoop;
  nElemPhMat  = cPrm->Nmom  * cPrm->locV3;

  printfQuda("%s: Memory report before Allocations", __func__);
  printMemoryInfo();

  dataPos = static_cast<complex<Float>*>(calloc(nElemPosLoc, SizeCplxFloat));
  if(dataPos  == NULL) errorQuda("%s: Could not allocate buffer: dataPos\n", __func__);
  
  if(cPrm->doMomProj){
    //- Allocate host data buffers
    dataMom_bcast = static_cast<complex<Float>*>(calloc(nElemMomTot, SizeCplxFloat));
    dataMom_h     = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
    dataMom       = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
    
    if(dataMom_bcast == NULL) errorQuda("%s: Could not allocate buffer: dataMom_bcast\n", __func__);
    if(dataMom_h     == NULL) errorQuda("%s: Could not allocate buffer: dataMom_h\n", __func__);
    if(dataMom       == NULL) errorQuda("%s: Could not allocate buffer: dataMom\n", __func__);
  }
  
  printfQuda("%s: Host buffers allocated\n", __func__);
  //------------------------------
  
  //- Allocate device data buffers

  //- That's the device loop-trace data buffer, always needed!
  cudaMalloc((void**)&(dataPos_d), SizeCplxFloat*nElemPosLoc);
  checkCudaError();
  cudaMemset(dataPos_d, 0, SizeCplxFloat*nElemPosLoc);

  if(cPrm->doMomProj){
    cudaMalloc( (void**)&(phaseMatrix_d), SizeCplxFloat*nElemPhMat);
    checkCudaError();
    cudaMemset(phaseMatrix_d, 0, SizeCplxFloat*nElemPhMat);
    
    cudaMalloc((void**)&(dataMom_d), SizeCplxFloat*nElemMomLoc);
    checkCudaError();
    cudaMemset(dataMom_d, 0, SizeCplxFloat*nElemMomLoc);

    cudaMalloc((void**)&(dataPosMP_d), SizeCplxFloat*nElemPosLoc);
    checkCudaError();
    cudaMemset(dataPosMP_d, 0, SizeCplxFloat*nElemPosLoc);
  }

  printfQuda("%s: Memory report after Allocations", __func__);
  printMemoryInfo();

  printfQuda("%s: Device buffers allocated\n", __func__);
}


// That's just a wrapper to copy the Gamma-matrix coefficient structure to constant memory
template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::copyGammaToConstMem(){
  copyGammaCoeffStructToSymbol<Float>();
  if(cPrm->doMomProj) copyGammaMapStructToSymbol<Float>();
  printfQuda("%s: Gamma utility structures copied to constant memory\n", __func__);
}


// Wrapper to create the Phase Matrix on GPU
template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::createPhaseMatrix(){
  createPhaseMatrixGPU<Float>(phaseMatrix_d, cPrm->momMatrix,
  			      cPrm->locV3, cPrm->Nmom, (int)cPrm->FTSign,
			      cPrm->localL, cPrm->totalL);
  
  printfQuda("%s: Phase matrix created\n", __func__);
}


template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::freeDataMemory(){

  printfQuda("%s: Memory report before freeing", __func__);
  printMemoryInfo();

  
  if(dataMom_bcast){
    free(dataMom_bcast);
    dataMom_bcast = nullptr;
  }
  if(dataMom_h){
    free(dataMom_h);
    dataMom_h = nullptr;
  }
  if(dataMom){
    free(dataMom);
    dataMom = nullptr;
  }
  if(dataPos){
    free(dataPos);
    dataPos = nullptr;
  }  
  printfQuda("%s: Host buffers freed\n", __func__);
  //------------------------------

  if(dataPos_d){
    cudaFree(dataPos_d);
    dataPos_d = nullptr;
  }
  if(dataPosMP_d){
    cudaFree(dataPosMP_d);
    dataPosMP_d = nullptr;
  }
  if(dataMom_d){
    cudaFree(dataMom_d);
    dataMom_d = nullptr;
  }

  if(phaseMatrix_d){
    cudaFree(phaseMatrix_d);
    phaseMatrix_d = nullptr;
  }
  printfQuda("%s: Device buffers freed\n", __func__);
  //------------------------------

  printfQuda("%s: Memory report after freeing", __func__);
  printMemoryInfo();
}


template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::printLoopComputeParams(){

  
  printfQuda("******************************************\n");
  printfQuda("    Parameters of the Loop Computation\n");
  printfQuda("Precision is %s\n", typeid(Float) == typeid(float) ? "single" : "double");
  printfQuda("Will%s use Multigrid\n", eigsolve->useMGenv ? "" : " NOT");
  printfQuda("Working with %s operators/fields\n", eigsolve->computeCoarse ? "coarse" : "fine");  
  printfQuda("Will%s perform Momentum Projection (Fourier Transform)\n", cPrm->doMomProj ? "" : " NOT");
  if(cPrm->doMomProj){
    printfQuda("Momentum Projection will be performed on GPU using cuBlas\n");
    printfQuda("Number of momenta: %d\n", cPrm->Nmom);
    printfQuda("Fourier transform Exp. Sign: %d\n", (int) cPrm->FTSign);
  }
  printfQuda("Will%s perform loop on non-local currents\n", cPrm->doNonLocal ? "" : " NOT");
  if(cPrm->doNonLocal){
    printfQuda("Will perform ultra-local loop, plus the following %d displacement entries:\n", cPrm->nDispEntries);
    for(int id=0;id<cPrm->nDispEntries;id++){
      char dispStr_c[cPrm->dispString.at(id).size()+1];
      strcpy(dispStr_c, cPrm->dispString.at(id).c_str());      

      if(cPrm->dispStop.at(id) == cPrm->dispStart.at(id))
	printfQuda("  %d: %s with length %d, #loops = %d, loop-offset = %d\n", id,
		   dispStr_c, cPrm->dispStart.at(id), cPrm->nLoopPerEntry.at(id), cPrm->nLoopOffset.at(id));
      else
	printfQuda("  %d: %s with lengths from %d to %d, #loops = %d, loop-offset = %d\n", id,
		   dispStr_c, cPrm->dispStart.at(id),cPrm->dispStop.at(id), cPrm->nLoopPerEntry.at(id), cPrm->nLoopOffset.at(id));
    }   
  }
  printfQuda("Total number of Loop Traces to perform: %d\n", cPrm->nLoop);
  printfQuda("Local  lattice size (x,y,z,t): %d %d %d %d \n", cPrm->localL[0], cPrm->localL[1], cPrm->localL[2], cPrm->localL[3]);
  printfQuda("Global lattice size (x,y,z,t): %d %d %d %d \n", cPrm->totalL[0], cPrm->totalL[1], cPrm->totalL[2], cPrm->totalL[3]);
  printfQuda("Global time extent: %d\n", cPrm->totT);
  printfQuda("Local  time extent: %d\n", cPrm->locT);
  printfQuda("Local  volume: %lld\n", cPrm->locV4);
  printfQuda("Local  3d volume: %lld\n", cPrm->locV3);
  printfQuda("Global 3d volume: %lld\n", cPrm->totV3);
  printfQuda("Transverse shift max. depth (not applicable now): %d\n", cPrm->max_depth);
  printfQuda("******************************************\n");
  
}


template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::prolongateEvec(ColorSpinorField *fineEvec, ColorSpinorField *coarseEvec){

  //- Multiple check layers
  if(!eigsolve->useMGenv) errorQuda("%s: This function is applicable only when using MG environment\n", __func__);
  if(!eigsolve->computeCoarse) errorQuda("%s: Not supposed to be called when computeCoarse is False\n", __func__);
  if(fieldOrder != QUDA_FLOAT2_FIELD_ORDER) errorQuda("%s: Vector prolongation requires fieldOrder = FLOAT2\n", __func__);
  
  MG_Mugiq &mg_env = *(eigsolve->getMGEnv());
  
  //- Create one fine and N_coarse temporary coarse fields
  //- Will be used for prolongating the coarse eigenvectors back to the fine lattice  
  std::vector<ColorSpinorField *> tmpCSF;
  ColorSpinorParam csParam(*refVec);
  QudaPrecision coarsePrec = coarseEvec->Precision();
  csParam.location = QUDA_CUDA_FIELD_LOCATION;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(coarsePrec);

  tmpCSF.push_back(ColorSpinorField::Create(csParam)); //- tmpCSF[0] is a fine field
  for(int lev=0;lev<mg_env.nCoarseLevels;lev++){
    tmpCSF.push_back(tmpCSF[lev]->CreateCoarse(mg_env.mgParams->geo_block_size[lev],
                                               mg_env.mgParams->spin_block_size[lev],
                                               mg_env.mgParams->n_vec[lev],
                                               coarsePrec,
                                               mg_env.mgParams->setup_location[lev+1]));
  }//-lev
    
  //- Prolongate the coarse eigenvectors recursively
  //- to get to the finest level
  *(tmpCSF[mg_env.nCoarseLevels]) = *coarseEvec;
  for(int lev=mg_env.nCoarseLevels;lev>1;lev--){
    blas::zero(*tmpCSF[lev-1]);
    if(!mg_env.transfer[lev-1]) errorQuda("%s: Transfer operator for level %d does not exist!\n", __func__, lev);
    mg_env.transfer[lev-1]->P(*(tmpCSF[lev-1]), *(tmpCSF[lev]));
  }
  blas::zero(*fineEvec);
  if(!mg_env.transfer[0]) errorQuda("%s: Transfer operator for finest level does not exist!\n", __func__);
  mg_env.transfer[0]->P(*fineEvec, *(tmpCSF[1]));

  for(int i=0;i<static_cast<int>(tmpCSF.size());i++) delete tmpCSF[i];

  printfQuda("%s: Vector prolongated\n", __func__);
}


template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::performMomentumProjection(){

  if(MomProjDone) errorQuda("%s: Not supposed to be called more than once!!", __func__);

  if(!commsAreSet) setupComms();
  
  const long long locV3 = cPrm->locV3;
  const int locT  = cPrm->locT;
  const int Nmom  = cPrm->Nmom;
  const int nLoop = cPrm->nLoop;
  const int nData = cPrm->nData;
  
  //-Some checks
  if(nData != nLoop*N_GAMMA_) errorQuda("%s: This function assumes that nData = nLoop * NGamma\n", __func__);

  
  /** 1. Convert indices from volume4d-inside-gamma-inside-Ndata to time-inside-Ndata-inside-volumeXYZ
   *  2. Map gamma matrices from G -> g5*G
   *  Ndata order is: gamma-inside-nloop in both cases
   */
  convertIdxOrder_mapGamma<Float>(dataPosMP_d, dataPos_d,
				  cPrm->nData, cPrm->nLoop, cPrm->nParity, cPrm->volumeCB, cPrm->localL);


  /** Perform momentum projection
   *-----------------------------
   * All matrices are stored on the device
   * Matrix dimensions in cuBlas are set such that the matrices are in column-major format as shown below
   *
   * Matrix Multiplication is: dataMom = dataPos * PhaseMatrix.
   *  dataPosMP_d   = (locT*nData,locV3) : input: loop-trace matrix with shuffled(converted) indices
   *  phaseMatrix_d = (locV3,Nmom)       : input: phase matrix
   *  dataMom_d     = (locT*nData,Nmom)  : output: momentum-projected data in column-major format (device)
   */  
  
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  complex<Float> al = complex<Float>{1.0,0.0};
  complex<Float> be = complex<Float>{0.0,0.0};

  if(typeid(Float) == typeid(double)){
    stat = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, locT*nData, Nmom, locV3,
                       (cuDoubleComplex*)&al,
		       (cuDoubleComplex*)dataPosMP_d, locT*nData,
		       (cuDoubleComplex*)phaseMatrix_d, locV3,
		       (cuDoubleComplex*)&be,
                       (cuDoubleComplex*)dataMom_d, locT*nData);
  }
  else if(typeid(Float) == typeid(float)){
    stat = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, locT*nData, Nmom, locV3,
                       (cuComplex*)&al,
		       (cuComplex*)dataPosMP_d, locT*nData,
		       (cuComplex*)phaseMatrix_d, locV3,
		       (cuComplex*)&be,
                       (cuComplex*)dataMom_d, locT*nData);
  }
  else errorQuda("%s: Precision not supported!\n", __func__);

  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: Momentum projection failed!\n", __func__);  

  
  //- extract the result from device (GPU) to host (CPU)
  stat = cublasGetMatrix(locT*nData, Nmom, sizeof(complex<Float>), dataMom_d, locT*nData, dataMom_h, locT*nData);
  if(stat != CUBLAS_STATUS_SUCCESS) errorQuda("%s: cuBlas data copying to CPU failed!\n", __func__);
  // ---------------------------------------------------------------------------------------

  
  /** Perform reduction over all processes
   * -------------------------------------
   * Create separate communicators
   * All processes with the same comm_coord(3) belong to COMM_SPACE communicator.
   * When performing the reduction over the COMM_SPACE communicator, the global sum
   * will be performed across all processes with the same time-coordinate,
   * and the result will be placed at the "root" of each of the "time" groups.
   * This means that the global result will exist only at the "time" processes, where each will
   * hold the sum for its corresponing time slices.
   * (In the case where only the time-direction is partitioned, MPI_Reduce is essentially a memcpy).
   */
  MPI_Datatype dataTypeMPI;
  if     ( typeid(Float) == typeid(float) ) dataTypeMPI = MPI_COMPLEX;
  else if( typeid(Float) == typeid(double)) dataTypeMPI = MPI_DOUBLE_COMPLEX;
  
  MPI_Reduce(dataMom_h, dataMom, nElemMomLoc, dataTypeMPI, MPI_SUM, 0, COMM_SPACE);
  
  
  /**
   * Then a Gathering is required, in order to put the global result from each of the "time" processes
   * into the final buffer (dataMom_bcast). This gathering must take place only across the "time" processes,
   * therefore another communicator involving only these processes must be created (COMM_TIME).
   * Finally, we need to Broadcast the final result to ALL processes, such that it is accessible to all of them.
   *
   * The final buffer follows order time-inside-gamma-inside-nLoops-inside-Mom:
   *              t + locT*ig + locT*nGamma*iL + locT*nGamma*nLoops*im = t + locT*id + locT*nData*im, where
   *    id = ig + nGamma*iL
   *    nData = nGamma*nLoops
   */
  MPI_Gather(dataMom      , nElemMomLoc, dataTypeMPI,
             dataMom_bcast, nElemMomLoc, dataTypeMPI,
             0, COMM_TIME);
  
  MPI_Bcast(dataMom_bcast, nElemMomTot, dataTypeMPI, 0, MPI_COMM_WORLD);

  
  //-- cleanup & return
  MPI_Comm_free(&COMM_SPACE);
  MPI_Comm_free(&COMM_TIME);

  cublasDestroy(handle);

  MomProjDone = MUGIQ_BOOL_TRUE;
}


//- This is the function that actually performs the trace
//- It's a public function, and it's called from the interface
template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::computeCoarseLoop(){

  int nEv = eigsolve->eigParams->nEv; // Number of eigenvectors

  //- Create a fine field, this will hold the prolongated version of each eigenvector
  ColorSpinorParam csParam(*refVec);
  printfQuda("%s: Field location (field,param): (%s,%s)\n", __func__,
	     refVec->Location() == 1 ? "CPU" : "GPU",
	     csParam.location == 1 ? "CPU" : "GPU");
  QudaPrecision evecPrec = eigsolve->eVecs[0]->Precision();
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(evecPrec);
  ColorSpinorField *fineEvecL = ColorSpinorField::Create(csParam);
  ColorSpinorField *fineEvecR = ColorSpinorField::Create(csParam);
  
  for(int id=-1;id<cPrm->nDispEntries;id++){
    char *dispEntry_c = nullptr;
    if( cPrm->doNonLocal && (id != -1) ){
      dispEntry_c = (char*)malloc(sizeof(char)*(cPrm->dispEntry.at(id).size()+1));
      strcpy(dispEntry_c, cPrm->dispEntry.at(id).c_str());
      printfQuda("\n\n%s: Will perform loop for displacement entry %s\n", __func__, dispEntry_c);
      displace->setupDisplacement(cPrm->dispString.at(id));
    }
    else printfQuda("\n\n%s: Will Run for ultra-local currents (displacement = 0)\n", __func__);    

    long long bufOffset;
    size_t bufByteSize;
    if( cPrm->doNonLocal && (id != -1) ){
      bufOffset = nElemPosLocPerLoop*cPrm->nLoopOffset.at(id); //- Jump the ultra-local plus loops of previous entry
      bufByteSize = SizeCplxFloat*nElemPosLocPerLoop*cPrm->nLoopPerEntry.at(id); //- #elem/entry 
    }
    else{
      bufOffset = 0;
      bufByteSize = SizeCplxFloat*nElemPosLocPerLoop;
    }

    cudaMemset(&(dataPos_d[bufOffset]), 0, bufByteSize);
    
    for(int n=0;n<nEv;n++){
      Float sigma = (Float)(*(eigsolve->eVals_sigma))[n];
      printfQuda("%s: Performing Loop trace for EV[%04d] = %+.16e\n", __func__, n, sigma);
      
      if(eigsolve->computeCoarse) prolongateEvec(fineEvecL, eigsolve->eVecs[n]);
      else *fineEvecL = *(eigsolve->eVecs[n]);

      if( cPrm->doNonLocal && (id != -1) ){
	//- Perform Displacements
	*fineEvecR = *fineEvecL; //- reset right vector to the original, un-displaced eigenvector
	int dispCount = 0;
	for(int idisp=1;idisp<=cPrm->dispStop.at(id);idisp++){
	  displace->doVectorDisplacement(DISPLACE_TYPE_COVARIANT, fineEvecR, idisp);
	  if(idisp >= cPrm->dispStart.at(id) && idisp <= cPrm->dispStop.at(id)){
	    long long dispOffset = nElemPosLocPerLoop*dispCount;
	    performLoopContraction<Float, fieldOrder>(&(dataPos_d[bufOffset+dispOffset]), fineEvecL, fineEvecR, sigma);
	    printfQuda("%s: EV[%04d] Loop trace for displacement = %02d completed\n", __func__, n, idisp);
	    dispCount++;
	  }
	}//-for displacement
      }
      else{
	//- Ultra-local
	*fineEvecR = *fineEvecL;
	performLoopContraction<Float, fieldOrder>(dataPos_d, fineEvecL, fineEvecR, sigma);
	printfQuda("%s: EV[%04d] - Loop trace for Ultra-local completed\n", __func__, n);
      }

    } //- Eigenvectors

    if(dispEntry_c) free(dispEntry_c);
  }//- Loop over displace entries

  //-Always copy the device position-space buffer to the host
  cudaMemcpy(dataPos, dataPos_d, SizeCplxFloat*nElemPosLoc, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  checkCudaError();
  printfQuda("\n%s: Device position-space data copied to host\n", __func__);
  
  if(cPrm->doMomProj){
    performMomentumProjection();
    printfQuda("\n%s: Momentum projection for all loops completed\n\n", __func__);
  }
  
  delete fineEvecL;
  delete fineEvecR;
  
}


//- Write the momentum-space loop data in HDF5 format
template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::writeLoopsHDF5_Mom(){

  if(!commsAreSet) setupComms();
  
  //- Only the "time" processes will write, they are the ones that have the globally reduced data buffer!
  if(IamTimeProcess){

    const int locT   = cPrm->locT;
    const int nGamma = cPrm->nG;
    const int nLoop  = cPrm->nLoop;
    
    //- Determine the data type for writing
    hid_t H5_dataType;
    if( typeid(Float) == typeid(float) ){
      H5_dataType = H5T_NATIVE_FLOAT;
      printfQuda("%s: Will write loop data in single precision\n", __func__);
    }
    else if( typeid(Float) == typeid(double)){
      H5_dataType = H5T_NATIVE_DOUBLE;
      printfQuda("%s: Will write loop data in double precision\n", __func__);
    }
    else errorQuda("%s: Precision not supported!\n", __func__);

    
    char filename_c[momSpaceFilename.size()+1];
    strcpy(filename_c, momSpaceFilename.c_str());
    printfQuda("%s: Momentum-space loop data HDF5 filename: %s\n", __func__, filename_c);
    
    const int dSetDim = 2; //- Size of each dataset (Time, real-imag)

    //- Start point (offset) for each process, in each dimension
    hsize_t start[dSetDim] = {static_cast<hsize_t>(tCoord*cPrm->localL[3]), 0};

    // Dimensions of the dataspace
    hsize_t tdims[dSetDim] = {static_cast<hsize_t>(cPrm->totalL[3]), 2}; // Global
    hsize_t ldims[dSetDim] = {static_cast<hsize_t>(cPrm->localL[3]), 2}; // Local

    //- Open the file
    hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);

    //H5Pset_fapl_mpio(fapl_id, COMM_TIME, MPI_INFO_NULL);
    H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL);
    hid_t file_id = H5Fcreate(filename_c, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    if(file_id<0) errorQuda("%s: Cannot open filename %s. Check that directory exists!\n", __func__, filename_c);
    H5Pclose(fapl_id);    

    int dStart = 0, dStop = 0;

    //- Begin creating the groups
    for(int im=0;im<cPrm->Nmom;im++){
      //-Momenta group
      char group1_tag[16];
      snprintf(group1_tag, sizeof(group1_tag), "mom_%+d_%+d_%+d",
	       cPrm->momMatrix[MOM_MATRIX_IDX(0,im)],
	       cPrm->momMatrix[MOM_MATRIX_IDX(1,im)],
	       cPrm->momMatrix[MOM_MATRIX_IDX(2,im)]);
      hid_t group1_id = H5Gcreate(file_id, group1_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      int iL = 0;
      for(int iDE=-1;iDE<cPrm->nDispEntries;iDE++){
	if(iDE == -1){
	  dStart = 0;
	  dStop  = 0;
	}
	else{
	  dStart = cPrm->dispStart.at(iDE);
	  dStop  = cPrm->dispStop.at(iDE);
	}
	for(int idisp=dStart;idisp<=dStop;idisp++){
	  //- Displacement group
	  char group2_tag[10];
	  if(iDE==-1){
	    snprintf(group2_tag,sizeof(group2_tag),"disp_0");
	  }
	  else{
	    std::string dStr = cPrm->dispString.at(iDE);
	    char disp_c[dStr.size()+1];
	    strcpy(disp_c, dStr.c_str());
	    snprintf(group2_tag,sizeof(group2_tag),"disp_%s_%d", disp_c, idisp);
	  }  
	  hid_t group2_id = H5Gcreate(group1_id, group2_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	  
	  for(int ig=0;ig<N_GAMMA_;ig++){
	    //- Gamma matrix group
	    std::string gStr = GammaName(ig);
	    char group3_tag[gStr.size()+1];
	    strcpy(group3_tag, gStr.c_str());
	    hid_t group3_id = H5Gcreate(group2_id, group3_tag, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	    //- Create filespaces and hyperslab
            hid_t h5_filespace = H5Screate_simple(dSetDim, tdims, NULL);
            hid_t dataset_id   = H5Dcreate(group3_id, "loop", H5_dataType, h5_filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            hid_t h5_subspace  = H5Screate_simple(dSetDim, ldims, NULL);
            h5_filespace = H5Dget_space(dataset_id);
            H5Sselect_hyperslab(h5_filespace, H5S_SELECT_SET, start, NULL, ldims, NULL);
            hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
            H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

	    const long long loopIdx = locT*ig + locT*nGamma*iL + locT*nGamma*nLoop*im;

	    //	    for(int t=0;t<locT;t++)
	    //	      printfQuda("%+16.e %16.e\n", dataMom[t+loopIdx].real(), dataMom[t+loopIdx].imag());
	    
	    herr_t status = H5Dwrite(dataset_id, H5_dataType, h5_subspace, h5_filespace, plist_id, &(dataMom[loopIdx]));
	    if(status<0) errorQuda("%s: Could not write data for (mom,disp,gamma) = (%d,%d,%d)\n", __func__,im,iL,ig);

            H5Sclose(h5_subspace);
            H5Dclose(dataset_id);
            H5Sclose(h5_filespace);
            H5Pclose(plist_id);

	    H5Gclose(group3_id);
	  }//- for gamma
	  
	  iL++;
	  H5Gclose(group2_id);
	} //- for dispStart-dispStop
      }//- for displacement entries
      
      H5Gclose(group1_id);
    }//- for momenta

    H5Fclose(file_id);
    
  }//- If time process
    
}


//- Write the position-space loop data in HDF5 format
template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::writeLoopsHDF5_Pos(){ 
  errorQuda("%s: Not supported yet!\n", __func__);
}


//- Public wrapper for writing the loops in HDF5 format
//- (called from the interface)
template <typename Float, QudaFieldOrder fieldOrder>
void Loop_Mugiq<Float, fieldOrder>::writeLoopsHDF5(){

  if(cPrm->doMomProj){
    if(writeDataMom) printfQuda("%s: Will write the momentum-space loop data in HDF5 format\n", __func__);
    else{
      warningQuda("%s: Performed momentum projection, but got writeDatMom = FALSE.\n", __func__);
      warningQuda("%s: Will proceed to write momentum-space loop data\n", __func__);
      writeDataMom = MUGIQ_BOOL_TRUE;
    }
    writeLoopsHDF5_Mom();
  }
  else{
    if(!writeDataPos){
      warningQuda("%s: Did not perform momentum projection, but got writeDatPos = FALSE.\n", __func__);
      warningQuda("%s: Will proceed to write position-space loop data\n", __func__);
      writeDataPos = MUGIQ_BOOL_TRUE;
    }
  }
  
  if(writeDataPos){
    printfQuda("%s: Will write the position-space loop data in HDF5 format\n", __func__);
    writeLoopsHDF5_Pos();
  }
  
}

//- Explicit instantiation of the templates of the Loop_Mugiq class
//- float and double will be the only typename templates that support is required,
//- so this is a 'feature' rather than a 'bug'
template class Loop_Mugiq<float, QUDA_FLOAT2_FIELD_ORDER>;
template class Loop_Mugiq<float, QUDA_FLOAT4_FIELD_ORDER>;
template class Loop_Mugiq<double, QUDA_FLOAT2_FIELD_ORDER>;
template class Loop_Mugiq<double, QUDA_FLOAT4_FIELD_ORDER>;
