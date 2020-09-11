#include <mugiq_contract_kernels.cuh>


//- Shared memory buffer with the left and right vectors, as well as the contraction data
template <typename Float>
extern __shared__ complex<Float> shmemBuf[];


//- Function that casts the __constant__ memory variable containing the gamma Coefficients
//- to its structure type, GammaCoeff
template <typename Float>
inline __device__ const GammaCoeff<Float>* gCoeff() {
  return reinterpret_cast<const GammaCoeff<Float>*>(cGamma);
}


template <typename Float>
__global__ void loopContract_kernel(complex<Float> *loopData, LoopContractArg<Float> *arg){




}


template __global__ void loopContract_kernel<float> (complex<float>  *loopData, LoopContractArg<float>  *arg);
template __global__ void loopContract_kernel<double>(complex<double> *loopData, LoopContractArg<double> *arg);
