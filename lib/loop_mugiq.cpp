#include <loop_mugiq.h>

template <typename Float>
Loop_Mugiq<Float>::Loop_Mugiq(MugiqLoopParam *loopParams_,
			      Eigsolve_Mugiq *eigsolve_) :
  cPrm(nullptr),
  dSt(nullptr),
  eigsolve(eigsolve_),
  dataPos_d(nullptr),
  dataPos_h(nullptr),
  dataMom_d(nullptr),
  dataMom_h(nullptr),
  dataMom_gs(nullptr),
  dataMom(nullptr),
  nElemMomTot(0),
  nElemMomLoc(0)
{
  printfQuda("\n*************************************************\n");
  printfQuda("%s: Creating Loop computation environment\n", __func__);
  
  cPrm = new LoopComputeParam(loopParams_, eigsolve->mg_env->mg_solver->B[0]);
  if(cPrm->doNonLocal) dSt = new LoopDispState<Float>(loopParams_);

  allocateDataMemory();
  copyGammaToConstMem();
  if(cPrm->doMomProj) createPhaseMatrix();
  
  printfQuda("*************************************************\n\n");

  printLoopComputeParams();
}

  
template <typename Float>
Loop_Mugiq<Float>::~Loop_Mugiq(){

  freeDataMemory();

  if(cPrm->doNonLocal) delete dSt;
  delete cPrm;
}


template <typename Float>
void Loop_Mugiq<Float>::allocateDataMemory(){

  nElemMomTot = cPrm->Ndata * cPrm->Nmom * cPrm->totT;
  nElemMomLoc = cPrm->Ndata * cPrm->Nmom * cPrm->locT;
  nElemPosLoc = cPrm->Ndata * cPrm->locV4;
  nElemPhMat  = cPrm->Nmom  * cPrm->locV3;
 
  //- Allocate host data buffers
  dataMom    = static_cast<complex<Float>*>(calloc(nElemMomTot, SizeCplxFloat));
  dataMom_gs = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
  dataMom_h  = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
  
  if(dataMom    == NULL) errorQuda("%s: Could not allocate buffer: dataMom\n", __func__);
  if(dataMom_gs == NULL) errorQuda("%s: Could not allocate buffer: dataMom_gs\n", __func__);
  if(dataMom_h  == NULL) errorQuda("%s: Could not allocate buffer: dataMom_h\n", __func__);
  
  if(!cPrm->doMomProj){
    dataPos_h = static_cast<complex<Float>*>(calloc(nElemPosLoc, SizeCplxFloat));
    if(dataPos_h  == NULL) errorQuda("%s: Could not allocate buffer: dataPos_h\n", __func__);
  }
  
  printfQuda("%s: Host buffers allocated\n", __func__);
  //------------------------------
  
  //- Allocate device data buffers
  cudaMalloc((void**)&(dataPos_d), SizeCplxFloat*nElemPosLoc);
  checkCudaError();
  cudaMemset(dataPos_d, 0, SizeCplxFloat*nElemPosLoc);

  cudaMalloc((void**)&(dataMom_d), SizeCplxFloat*nElemMomLoc);
  checkCudaError();
  cudaMemset(dataMom_d, 0, SizeCplxFloat*nElemMomLoc);

  if(cPrm->doMomProj){
    cudaMalloc( (void**)&(phaseMatrix_d), SizeCplxFloat*nElemPhMat);
    checkCudaError();
    cudaMemset(phaseMatrix_d, 0, SizeCplxFloat*nElemPhMat);    
  }
  
  printfQuda("%s: Device buffers allocated\n", __func__);
  //------------------------------
  
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

  if(dataMom){
    free(dataMom);
    dataMom = nullptr;
  }
  if(dataMom_gs){
    free(dataMom_gs);
    dataMom_gs = nullptr;
  }
  if(dataMom_h){
    free(dataMom_h);
    dataMom_h = nullptr;
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
    printfQuda("Non-local path string: %s\n", cPrm->pathString);
    printfQuda("Non-local path length: %d\n", cPrm->pathLen);
  }    
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

  for(int im=0;im<cPrm->Nmom;im++){
    for(int id=0;id<cPrm->Ndata;id++){
      printfQuda("Loop for momentum (%+d,%+d,%+d), Gamma[%d]:\n",
		 cPrm->momMatrix[MOM_MATRIX_IDX(0,im)],
		 cPrm->momMatrix[MOM_MATRIX_IDX(1,im)],
		 cPrm->momMatrix[MOM_MATRIX_IDX(2,im)], id);
      for(int it=0;it<cPrm->totT;it++){
	//- FIXME: Check if loop Index is correct
	int loopIdx = id + cPrm->Ndata*it + cPrm->Ndata*cPrm->totT*im;
	printfQuda("%d %+.8e %+.8e\n", it, dataMom[loopIdx].real(), dataMom[loopIdx].imag());
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
void Loop_Mugiq<Float>::computeCoarseLoop(){

  int nEv = eigsolve->eigParams->nEv; // Number of eigenvectors

  //- Create a fine field, this will hold the prolongated version of each eigenvector
  ColorSpinorParam csParam(*(eigsolve->mg_env->mg_solver->B[0]));
  QudaPrecision coarsePrec = eigsolve->eVecs[0]->Precision();
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(coarsePrec);
  ColorSpinorField *fineEvecL = ColorSpinorField::Create(csParam);
  ColorSpinorField *fineEvecR = ColorSpinorField::Create(csParam);

  cudaMemset(dataPos_d, 0, SizeCplxFloat*nElemPosLoc);
  for(int n=0;n<nEv;n++){
    Float sigma = (Float)(*(eigsolve->eVals_sigma))[n];
    printfQuda("**************** %+.16e\n", sigma);

    if(eigsolve->computeCoarse) prolongateEvec(fineEvecL, eigsolve->eVecs[n]);
    else *fineEvecL = *(eigsolve->eVecs[n]);

    if(!cPrm->doNonLocal) *fineEvecR = *fineEvecL;
    else{
      //-Perform Shifts
      //- TODO: Probably not as simple as that, but not much more complicated either
      //      dSt->performShift(fineEvecR, fineEvecL);
      //- For now, just set them equal
      *fineEvecR = *fineEvecL;
    }

    performLoopContraction<Float>(dataPos_d, fineEvecL, fineEvecR);    

  } //- Eigenvectors

  
  delete fineEvecL;
  delete fineEvecR;
  
}


//- Explicit instantiation of the templates of the Loop_Mugiq class
//- float and double will be the only typename templates that support is required,
//- so this is a 'feature' rather than a 'bug'
template class Loop_Mugiq<float>;
template class Loop_Mugiq<double>;
