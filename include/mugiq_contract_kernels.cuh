#ifndef _MUGIQ_CONTRACT_KERNELS_CUH
#define _MUGIQ_CONTRACT_KERNELS_CUH

#include <contract_util.cuh>

using namespace quda;

template <typename Float>
__global__ void phaseMatrix_kernel(complex<Float> *phaseMatrix, int *momMatrix, MomProjArg *arg);


template <typename Float>
__global__ void loopContract_kernel(complex<Float> *loopData_d,  LoopContractArg<Float> *arg);

#endif // _MUGIQ_CONTRACT_KERNELS_CUH
