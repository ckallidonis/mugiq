#ifndef _MUGIQ_DISPLACE_KERNELS_CUH
#define _MUGIQ_DISPLACE_KERNELS_CUH

#include <contract_util.cuh>

using namespace quda;

template <typename Float, typename Arg>
__global__ void covariantDisplacementVector_kernel(Arg *arg, DisplaceDir dispDir, DisplaceSign dispSign);


#endif // _MUGIQ_DISPLACE_KERNELS_CUH
