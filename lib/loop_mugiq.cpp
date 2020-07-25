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
  
  printfQuda("*************************************************\n\n");
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

  for(int im=0;im<cPrm->Nmom;im++){
    for(int id=0;id<cPrm->Ndata;id++){
      printfQuda("Loop for momentum (%+d,%+d,%+d), Gamma[%d]:\n",
		 cPrm->momMatrix[im][0],
		 cPrm->momMatrix[im][1],
		 cPrm->momMatrix[im][2], id);
      for(int it=0;it<cPrm->totT;it++){
	int loopIdx = id + cPrm->Ndata*it + cPrm->Ndata*cPrm->totT*im;
	printfQuda("%d %+.8e %+.8e\n", it, dataMom[loopIdx].real(), dataMom[loopIdx].imag());
      }
    }
  }
  
}


template <typename Float>
void Loop_Mugiq<Float>::computeCoarseLoop(){
  
}


//- Explicit instantiation of the templates of the Loop_Mugiq class
//- float and double will be the only typename templates that support is required,
//- so this is a 'feature' rather than a 'bug'
template class Loop_Mugiq<float>;
template class Loop_Mugiq<double>;
