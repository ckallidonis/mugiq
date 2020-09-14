#ifndef _MUGIQ_UTIL_KERNELS_CUH
#define _MUGIQ_UTIL_KERNELS_CUH

#include <contract_util.cuh>

using namespace quda;

template <typename Float>
__global__ void phaseMatrix_kernel(complex<Float> *phaseMatrix, int *momMatrix, MomProjArg *arg);


template <typename Float>
__global__ void convertIdxMomProj_kernel(complex<Float> *dataOut, const complex<Float> *dataIn, ConvertIdxArg *arg);


#endif // _MUGIQ_UTIL_KERNELS_CUH
