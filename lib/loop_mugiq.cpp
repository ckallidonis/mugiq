#include <loop_mugiq.h>
#include <cublas_v2.h>
#include <mpi.h>

template <typename Float>
Loop_Mugiq<Float>::Loop_Mugiq(MugiqLoopParam *loopParams_,
			      Eigsolve_Mugiq *eigsolve_) :
  cPrm(nullptr),
  displace(nullptr),
  eigsolve(eigsolve_),
  dataPos_d(nullptr),
  dataPosMP_d(nullptr),
  dataMom_d(nullptr),
  dataPos_h(nullptr),
  dataMom_h(nullptr),
  dataMom(nullptr),
  dataMom_bcast(nullptr),
  nElemMomTot(0),
  nElemMomLoc(0),
  MomProjDone(MUGIQ_BOOL_FALSE)
{
  printfQuda("\n*************************************************\n");
  printfQuda("%s: Creating Loop computation environment\n", __func__);
  
  cPrm = new LoopComputeParam(loopParams_, eigsolve->mg_env->mg_solver->B[0]);
  if(cPrm->doNonLocal) displace = new Displace<Float>(loopParams_,
						      eigsolve->mg_env->mg_solver->B[0],
						      eigsolve->eVecs[0]->Precision());

  allocateDataMemory();
  copyGammaToConstMem();
  if(cPrm->doMomProj) createPhaseMatrix();
  
  printfQuda("*************************************************\n\n");

  printLoopComputeParams();
}

  
template <typename Float>
Loop_Mugiq<Float>::~Loop_Mugiq(){

  freeDataMemory();

  if(cPrm->doNonLocal) delete displace;
  delete cPrm;
}


template <typename Float>
void Loop_Mugiq<Float>::allocateDataMemory(){

  nElemMomTotPerLoop = cPrm->nG * cPrm->Nmom * cPrm->totT;
  nElemMomLocPerLoop = cPrm->nG * cPrm->Nmom * cPrm->locT;
  nElemPosLocPerLoop = cPrm->nG * cPrm->locV4;
  
  nElemMomTot = nElemMomTotPerLoop * cPrm->nLoop;
  nElemMomLoc = nElemMomLocPerLoop * cPrm->nLoop;
  nElemPosLoc = nElemPosLocPerLoop * cPrm->nLoop;
  nElemPhMat  = cPrm->Nmom  * cPrm->locV3;

  printfQuda("%s: Memory report before Allocations", __func__);
  printMemoryInfo();
  
  if(cPrm->doMomProj){
    //- Allocate host data buffers
    dataMom_bcast = static_cast<complex<Float>*>(calloc(nElemMomTot, SizeCplxFloat));
    dataMom_h     = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
    dataMom       = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
    
    if(dataMom_bcast == NULL) errorQuda("%s: Could not allocate buffer: dataMom_bcast\n", __func__);
    if(dataMom_h     == NULL) errorQuda("%s: Could not allocate buffer: dataMom_h\n", __func__);
    if(dataMom       == NULL) errorQuda("%s: Could not allocate buffer: dataMom\n", __func__);
  }
  else{
    dataPos_h = static_cast<complex<Float>*>(calloc(nElemPosLoc, SizeCplxFloat));
    if(dataPos_h  == NULL) errorQuda("%s: Could not allocate buffer: dataPos_h\n", __func__);
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
template <typename Float>
void Loop_Mugiq<Float>::copyGammaToConstMem(){
  copyGammaCoeffStructToSymbol<Float>();
  printfQuda("%s: Gamma coefficient structure copied to constant memory\n", __func__);
}


// Wrapper to create the Phase Matrix on GPU
template <typename Float>
void Loop_Mugiq<Float>::createPhaseMatrix(){
  createPhaseMatrixGPU<Float>(phaseMatrix_d, cPrm->momMatrix,
  			      cPrm->locV3, cPrm->Nmom, (int)cPrm->FTSign,
			      cPrm->localL, cPrm->totalL);
  
  printfQuda("%s: Phase matrix created\n", __func__);
}


template <typename Float>
void Loop_Mugiq<Float>::freeDataMemory(){

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
  if(dataPos_h){
    free(dataPos_h);
    dataPos_h = nullptr;
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


template <typename Float>
void Loop_Mugiq<Float>::printLoopComputeParams(){

  
  printfQuda("******************************************\n");
  printfQuda("    Parameters of the Loop Computation\n");
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


template <typename Float>
void Loop_Mugiq<Float>::printData_ASCII(){
  errorQuda("%s: This function unsupported for now!\n");

  for(int im=0;im<cPrm->Nmom;im++){
    for(int id=0;id<cPrm->nData;id++){
      printfQuda("Loop for momentum (%+d,%+d,%+d), Gamma[%d]:\n",
		 cPrm->momMatrix[MOM_MATRIX_IDX(0,im)],
		 cPrm->momMatrix[MOM_MATRIX_IDX(1,im)],
		 cPrm->momMatrix[MOM_MATRIX_IDX(2,im)], id);
      for(int it=0;it<cPrm->totT;it++){
	//- FIXME: Check if loop Index is correct
	int loopIdx = id + cPrm->nData*it + cPrm->nData*cPrm->totT*im;
	printfQuda("%d %+.8e %+.8e\n", it, dataMom_bcast[loopIdx].real(), dataMom_bcast[loopIdx].imag());
      }
    }
  }
  
}



template <typename Float>
void Loop_Mugiq<Float>::prolongateEvec(ColorSpinorField *fineEvec, ColorSpinorField *coarseEvec){

  MG_Mugiq &mg_env = *(eigsolve->getMGEnv());
  
  //- Create one fine and N_coarse temporary coarse fields
  //- Will be used for prolongating the coarse eigenvectors back to the fine lattice
  std::vector<ColorSpinorField *> tmpCSF;
  ColorSpinorParam csParam(*(mg_env.mg_solver->B[0]));
  QudaPrecision coarsePrec = coarseEvec->Precision();
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
  
  //- Prolongate the coarse eigenvectors recursively to get
  //- to the finest level  
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
  
}


template <typename Float>
void Loop_Mugiq<Float>::performMomentumProjection(){

  if(MomProjDone)
    errorQuda("%s: Not supposed to be called more than once!!", __func__);

  const long long locV3 = cPrm->locV3;
  const int locT  = cPrm->locT;
  const int totT  = cPrm->totT;
  const int Nmom  = cPrm->Nmom;
  const int nLoop = cPrm->nLoop;
  const int nData = cPrm->nData;
  
  //-Some checks
  if(nData != nLoop*N_GAMMA_) errorQuda("%s: This function assumes that nData = nLoop * NGamma\n", __func__);
  
  /** Convert indices from volume4d-inside-gamma-inside-Ndata to time-inside-Ndata-inside-volumeXYZ
   *  AND
   *  Map gamma matrices from G -> g5*G
   *
   *  Ndata order is: gamma-inside-nloop in both cases
   */
  convertIdxOrder_mapGamma<Float>(dataPosMP_d, dataPos_d,
				  cPrm->nData, cPrm->nLoop, cPrm->nParity, cPrm->volumeCB, cPrm->localL);
  
  /** Perform momentum projection
   *-----------------------------
   * Matrix Multiplication Out = PH^T * In.
   *  phaseMatrix_dev=(locV3,Nmom) is the phase matrix in column-major format, its transpose is used for multiplication
   *  dataPosMP_d = (locV3,nData*locT) is the device input loop-trace matrix with shuffled(converted) indices
   *  dataMom_d = (Nmom,nData*locT) is the output matrix in column-major format (device)
   */  
    
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  complex<Float> al = complex<Float>{1.0,0.0};
  complex<Float> be = complex<Float>{0.0,0.0};

  if(typeid(Float) == typeid(double)){
    stat = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmom, nData*locT, locV3,
                       (cuDoubleComplex*)&al, (cuDoubleComplex*)phaseMatrix_d, locV3,
                       (cuDoubleComplex*)dataPosMP_d, locV3,
		       (cuDoubleComplex*)&be,
                       (cuDoubleComplex*)dataMom_d, Nmom);
  }
  else if(typeid(Float) == typeid(float)){
    stat = cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nmom, nData*locT, locV3,
                       (cuComplex*)&al, (cuComplex*)phaseMatrix_d, locV3,
                       (cuComplex*)dataPosMP_d, locV3,
		       (cuComplex*)&be,
                       (cuComplex*)dataMom_d, Nmom);
  }
  else errorQuda("%s: Precision not supported!\n", __func__);

  if(stat != CUBLAS_STATUS_SUCCESS)
    errorQuda("%s: Momentum projection failed!\n", __func__);  

  
  //-- extract the result from GPU to CPU
  stat = cublasGetMatrix(Nmom, nData*locT, sizeof(complex<Float>), dataMom_d, Nmom, dataMom_h, Nmom);
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
   *
   * Then a Gathering is required, in order to put the global result from each of the "time" processes
   * into the final buffer (dataMom_bcast). This gathering must take place only across the "time" processes,
   * therefore another communicator involving only these processes must be created (COMM_TIME).
   * Finally, we need to Broadcast the final result to ALL processes, such that it is accessible to all of them.
   *
   * The final buffer follows order Mom-inside-Gamma-inside-nLoops-inside-T:
   *              im + Nmom*ig + Nmom*nGamma*iL + Nmom*nGamma*nLoops*t = im + Nmom*id + Nmom*nData*t, where
   *    id = ig + nGamma*iL
   *    nData = nGamma*nLoops
   */

  MPI_Datatype dataTypeMPI;
  if     ( typeid(Float) == typeid(float) ) dataTypeMPI = MPI_COMPLEX;
  else if( typeid(Float) == typeid(double)) dataTypeMPI = MPI_DOUBLE_COMPLEX;

  //-- Create space-communicator
  int space_rank, space_size;
  MPI_Comm COMM_SPACE;
  int tCoord = comm_coord(3);
  int cRank = comm_rank();
  MPI_Comm_split(MPI_COMM_WORLD, tCoord, cRank, &COMM_SPACE);
  MPI_Comm_rank(COMM_SPACE,&space_rank);
  MPI_Comm_size(COMM_SPACE,&space_size);

  //-- Create time communicator
  int time_rank, time_size;
  int time_tag = 1000;
  MPI_Comm COMM_TIME;
  int time_color = comm_rank();   //-- Determine the "color" which distinguishes the "time" processes from the rest
  if( (comm_coord(0) == 0) &&
      (comm_coord(1) == 0) &&
      (comm_coord(2) == 0) ) time_color = (time_tag>comm_size()) ? time_tag : time_tag+comm_size();

  MPI_Comm_split(MPI_COMM_WORLD, time_color, tCoord, &COMM_TIME);
  MPI_Comm_rank(COMM_TIME,&time_rank);
  MPI_Comm_size(COMM_TIME,&time_size);

  
  MPI_Reduce(dataMom_h, dataMom, Nmom*nData*locT, dataTypeMPI, MPI_SUM, 0, COMM_SPACE);

  
  MPI_Gather(dataMom      , Nmom*nData*locT, dataTypeMPI,
             dataMom_bcast, Nmom*nData*locT, dataTypeMPI,
             0, COMM_TIME);

  
  MPI_Bcast(dataMom_bcast, Nmom*nData*totT, dataTypeMPI, 0, MPI_COMM_WORLD);

  
  //-- cleanup & return
  MPI_Comm_free(&COMM_SPACE);
  MPI_Comm_free(&COMM_TIME);

  cublasDestroy(handle);

  MomProjDone = MUGIQ_BOOL_TRUE;

}


//- This is the function that actually performs the trace
//- It's a public function, and it's called from the interface
template <typename Float>
void Loop_Mugiq<Float>::computeCoarseLoop(){

  int nEv = eigsolve->eigParams->nEv; // Number of eigenvectors

  //- Create a fine field, this will hold the prolongated version of each eigenvector
  ColorSpinorParam csParam(*(eigsolve->mg_env->mg_solver->B[0]));
  printfQuda("%s: Field location (field,param): (%s,%s)\n", __func__,
	     eigsolve->mg_env->mg_solver->B[0]->Location() == 1 ? "CPU" : "GPU",
	     csParam.location == 1 ? "CPU" : "GPU");
  QudaPrecision coarsePrec = eigsolve->eVecs[0]->Precision();
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(coarsePrec);
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

    cudaMemset(&dataPos_d[bufOffset], 0, bufByteSize);
    
    for(int n=0;n<nEv;n++){
      Float sigma = (Float)(*(eigsolve->eVals_sigma))[n];
      printfQuda("%s: Performing Loop trace for EV[%04d] = %+.16e\n", __func__, n, sigma);
      
      if(eigsolve->computeCoarse) prolongateEvec(fineEvecL, eigsolve->eVecs[n]);
      else *fineEvecL = *(eigsolve->eVecs[n]);

      if( cPrm->doNonLocal && (id != -1) ){
	//	displace->resetDispVec(fineEvecL); //- reset consecutively displaced vector to the original eigenvector[n]
	*fineEvecR = *fineEvecL; //- reset right vector to the original, un-displaced eigenvector
	int dispCount = 0;
	for(int idisp=1;idisp<=cPrm->dispStop.at(id);idisp++){
	  displace->doVectorDisplacement(DISPLACE_TYPE_COVARIANT, fineEvecR, idisp);
	  if(idisp >= cPrm->dispStart.at(id) && idisp <= cPrm->dispStop.at(id)){
	    long long dispOffset = nElemPosLocPerLoop*dispCount;
	    performLoopContraction<Float>(&dataPos_d[bufOffset+dispOffset], fineEvecL, fineEvecR, sigma);
	    printfQuda("%s: EV[%04d] Loop trace for displacement = %02d completed\n", __func__, n, idisp);
	    dispCount++;
	  }
	}//-for displacement
      }
      else{
	*fineEvecR = *fineEvecL;
	performLoopContraction<Float>(dataPos_d, fineEvecL, fineEvecR, sigma);
	printfQuda("%s: EV[%04d] - Loop trace for Ultra-local completed\n", __func__, n);
      }

    } //- Eigenvectors

    if(dispEntry_c) free(dispEntry_c);
  }//- Loop over displace entries

  
  if(cPrm->doMomProj){
    performMomentumProjection();
    printfQuda("\n%s: Momentum projection for all loops completed\n\n", __func__);
  }
  else{
    cudaMemcpy(dataPos_h, dataPos_d, SizeCplxFloat*nElemPosLoc, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkCudaError();
  }
  
  delete fineEvecL;
  delete fineEvecR;
  
}


//- Explicit instantiation of the templates of the Loop_Mugiq class
//- float and double will be the only typename templates that support is required,
//- so this is a 'feature' rather than a 'bug'
template class Loop_Mugiq<float>;
template class Loop_Mugiq<double>;
