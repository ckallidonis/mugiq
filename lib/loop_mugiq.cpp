#include <loop_mugiq.h>

template <typename Float>
Loop_Mugiq<Float>::Loop_Mugiq(MugiqLoopParam *loopParams_,
			      Eigsolve_Mugiq *eigsolve_) :
  trParams(nullptr),
  shifts(nullptr),
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
  
  trParams = new MugiqTraceParam(loopParams_, eigsolve->mg_env->mg_solver->B[0]);  

  allocateDataMemory(); 
  
}

  
template <typename Float>
Loop_Mugiq<Float>::~Loop_Mugiq(){

  freeDataMemory();

  delete trParams;
}


template <typename Float>
void Loop_Mugiq<Float>::allocateDataMemory(){

  nElemMomTot = trParams->Ndata * trParams->Nmom * trParams->totT;
  nElemMomLoc = trParams->Ndata * trParams->Nmom * trParams->locT;
  nElemPosLoc = trParams->Ndata * trParams->locV4;
  
  //- Allocate host data buffers
  dataMom    = static_cast<complex<Float>*>(calloc(nElemMomTot, SizeCplxFloat));
  dataMom_gs = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
  dataMom_h  = static_cast<complex<Float>*>(calloc(nElemMomLoc, SizeCplxFloat));
  
  if(dataMom    == NULL) errorQuda("%s: Could not allocate buffer: dataMom\n", __func__);
  if(dataMom_gs == NULL) errorQuda("%s: Could not allocate buffer: dataMom_gs\n", __func__);
  if(dataMom_h  == NULL) errorQuda("%s: Could not allocate buffer: dataMom_h\n", __func__);
  
  if(!trParams->doMomProj){
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

  printfQuda("%s: Device buffers allocated\n", __func__);
  //------------------------------
  
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
  printfQuda("%s: Device buffers freed\n", __func__);
  //------------------------------

}


template <typename Float>
void Loop_Mugiq<Float>::printData_ASCII(){

  for(int im=0;im<trParams->Nmom;im++){
    for(int id=0;id<trParams->Ndata;id++){
      printfQuda("Loop for momentum (%+d,%+d,%+d), Gamma[%d]:\n",
		 trParams->momMatrix[im][0],
		 trParams->momMatrix[im][1],
		 trParams->momMatrix[im][2], id);
      for(int it=0;it<trParams->totT;it++){
	int loopIdx = id + trParams->Ndata*it + trParams->Ndata*trParams->totT*im;
	printfQuda("%d %+.8e %+.8e\n", it, dataMom_h[loopIdx].real(), dataMom_h[loopIdx].imag());
      }
    }
  }
  
}


template <typename Float>
void Loop_Mugiq<Float>::createCoarseLoop_uLocal(){
  
  if(trParams->calcType == LOOP_CALC_TYPE_OPT_KERNEL)
    createCoarseLoop_uLocal_optKernel();
  else
    errorQuda("%s: Unsupported calculation type for coarseLoop_uLocal\n", __func__);
  
}


template <typename Float>
void Loop_Mugiq<Float>::createCoarseLoop_uLocal_optKernel(){




  

}


//- Explicit instantiation of the templates of the Loop_Mugiq class
//- float and double will be the only typename templates that support is required,
//- so this is a 'feature' rather than a 'bug'
template class Loop_Mugiq<float>;
template class Loop_Mugiq<double>;

