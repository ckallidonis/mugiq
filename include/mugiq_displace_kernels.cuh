#ifndef _MUGIQ_DISPLACE_KERNELS_CUH
#define _MUGIQ_DISPLACE_KERNELS_CUH

#include <contract_util.cuh>

using namespace quda;

template <typename Float>
__global__ void covariantDisplacementVector_kernel(CovDispVecArg<Float> *arg, DisplaceDir dispDir, DisplaceSign dispSign);


#endif // _MUGIQ_DISPLACE_KERNELS_CUH
