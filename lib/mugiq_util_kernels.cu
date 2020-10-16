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


//- Function that casts the __constant__ memory variable containing the gamma mapping info
//- to its structure type, GammaMap
template <typename Float>
inline __device__ const GammaMap<Float>* gMap() {
  return reinterpret_cast<const GammaMap<Float>*>(cGammaMap);
}

//- Wrapper to copy Gamma map structure to __constant__ memory
template <typename Float>
void copyGammaMaptoSymbol(GammaMap<Float> gmap_struct){
  cudaMemcpyToSymbol(cGammaMap, &gmap_struct, sizeof(GammaMap<Float>));
}

template void copyGammaMaptoSymbol(GammaMap<float> gmap_struct);
template void copyGammaMaptoSymbol(GammaMap<double> gmap_struct);


template <typename Float>
__global__ void convertIdxOrder_mapGamma_kernel(complex<Float> *dataOut, const complex<Float> *dataIn, ConvertIdxArg *arg){

  int x_cb = blockIdx.x*blockDim.x + threadIdx.x;  // checkerboard site within 4d local volume
  int pty  = blockIdx.y*blockDim.y + threadIdx.y;  // parity (even/odd)
  int ig   = blockIdx.z*blockDim.z + threadIdx.z;  // gamma-matrix index
  
  if(x_cb >= arg->volumeCB) return;
  if(pty  >= arg->nParity)  return;
  if(ig   >= N_GAMMA_) errorQuda("%s: Maximum z-block dimension must be %d", __func__, N_GAMMA_);

  int tid = x_cb + arg->volumeCB*pty; // full site index

  //- Local coordinates
  //- We need these to convert even-odd indexing to full lexicographic format
  int crd[5];
  getCoords(crd, x_cb, arg->localL, pty);
  int x = crd[0];
  int y = crd[1];
  int z = crd[2];
  int t = crd[3];

  int Lx = arg->localL[0];
  int Ly = arg->localL[1];
  int Lt = arg->localL[3];
  
  //- Get the gamma mapping info from constant memory
  const GammaMap<Float> *gammaMap = gMap<Float>();
  
  for(int iL=0;iL<arg->nLoop;iL++){
    int idataFrom = ig + N_GAMMA_*iL;
    int idxFrom = tid + arg->volumeCB*arg->nParity*idataFrom; //- Volume indices here are in even-odd format

    int idataTo = gammaMap->index[ig] + N_GAMMA_*iL; //- Convert gamma index from G -> g5*G
    int v3 = x + Lx*y + Lx*Ly*z; //- Volume indices of the output buffer are in the full-volume format    
    int idxTo = t + Lt*idataTo + Lt*arg->nData*v3;

    dataOut[idxTo] = gammaMap->sign[ig] * dataIn[idxFrom];
  }//- for loops
  
}

template __global__ void convertIdxOrder_mapGamma_kernel<float> (complex<float> *dataOut,  const complex<float> *dataIn,
								 ConvertIdxArg *arg);
template __global__ void convertIdxOrder_mapGamma_kernel<double>(complex<double> *dataOut, const complex<double> *dataIn,
								 ConvertIdxArg *arg);
//---------------------------------------------------------------------------
