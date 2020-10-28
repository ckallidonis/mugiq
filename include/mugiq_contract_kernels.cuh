#ifndef _MUGIQ_CONTRACT_KERNELS_CUH
#define _MUGIQ_CONTRACT_KERNELS_CUH

#include <contract_util.cuh>

using namespace quda;

template <typename Float>
void copyGammaCoefftoSymbol(GammaCoeff<Float> gcoeff_struct);

template <typename Float, typename Arg>
__global__ void loopContract_kernel(complex<Float> *loopData_d, Arg *arg);

#endif // _MUGIQ_CONTRACT_KERNELS_CUH
