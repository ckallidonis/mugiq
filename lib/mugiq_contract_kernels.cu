#include <mugiq_contract_kernels.cuh>

template <typename Float>
__global__ void loopContract_kernel(complex<Float> *loopData, LoopContractArg<Float> *arg){




}


template __global__ void loopContract_kernel<float> (complex<float>  *loopData, LoopContractArg<float>  *arg);
template __global__ void loopContract_kernel<double>(complex<double> *loopData, LoopContractArg<double> *arg);
