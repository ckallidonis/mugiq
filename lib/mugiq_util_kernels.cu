#include <mugiq_util_kernels.cuh>

template <typename Float>
__global__ void phaseMatrix_kernel(complex<Float> *phaseMatrix, int *momMatrix, MomProjArg *arg){

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(tid < arg->locV3){ // run through the spatial volume
    
    int lcoord[MOM_DIM_];
    int gcoord[MOM_DIM_];
    
    int a1 = tid / arg->localL[0];
    int a2 = a1 / arg->localL[1];
    lcoord[0] = tid - a1 * arg->localL[0];
    lcoord[1] = a1  - a2 * arg->localL[1];
    lcoord[2] = a2;
    
    gcoord[0] = lcoord[0] + arg->commCoord[0] * arg->localL[0];
    gcoord[1] = lcoord[1] + arg->commCoord[1] * arg->localL[1];
    gcoord[2] = lcoord[2] + arg->commCoord[2] * arg->localL[2];
    
    Float sgn = (Float) arg->FTSign;
    for(int im=0;im<arg->Nmom;im++){
      Float phase = 0.0;
      for(int id=0;id<arg->momDim;id++)
	phase += momMatrix[MOM_MATRIX_IDX(id,im)]*gcoord[id] / (Float)arg->totalL[id];
      
      phaseMatrix[tid + arg->locV3*im].x =     cos(2.0*PI*phase);
      phaseMatrix[tid + arg->locV3*im].y = sgn*sin(2.0*PI*phase);
    }
    
  }//-- tid check
  
}//--kernel

template __global__ void phaseMatrix_kernel<float> (complex<float>  *phaseMatrix, int *momMatrix, MomProjArg *arg);
template __global__ void phaseMatrix_kernel<double>(complex<double> *phaseMatrix, int *momMatrix, MomProjArg *arg);
//---------------------------------------------------------------------------
