//#include <mg_mugiq.h>
#include <eigsolve_mugiq.h>
//#include <interface_mugiq.h>
//#include <gamma.h>
//#include <loop_coarse_ulocal.h>
#include <loop_mugiq.h>
//#include <util_mugiq.h>

template <typename Float>
Loop_Mugiq<Float>::Loop_Mugiq(MugiqLoopParam *loopParams_,
			      Eigsolve_Mugiq *eigsolve_) :
  trParams(nullptr),
  eigsolve(eigsolve_),
  dataMom_h(nullptr),
  dataMom_d(nullptr),
  nElemMom(0),
  loopSizeMom(0)
{

  trParams = new MugiqTraceParam(loopParams_, eigsolve->mg_env->mg_solver->B[0]);
  
  nElemMom = trParams->Nmom * trParams->totT * trParams->Ndata;       //- Number of elements in momentum-space data buffers

  loopSizeMom = sizeof(complex<Float>) * nElemMom; //- Size of data buffers in bytes

  //- Allocate host buffer, needed always
  dataMom_h = static_cast<complex<Float>*>(malloc(loopSizeMom));
  if(dataMom_h == NULL) errorQuda("%s: Could not allocate host loop data buffer\n", __func__);
  memset(dataMom_h, 0, loopSizeMom);
  
}

template <typename Float>
Loop_Mugiq<Float>::~Loop_Mugiq(){

  if(dataMom_h) free(dataMom_h);
  dataMom_h = nullptr;

  delete trParams;
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

