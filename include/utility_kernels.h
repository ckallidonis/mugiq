#ifndef _UTILITY_KERNELS_H
#define _UTILITY_KERNELS_H

#include <kernels_mugiq.h>

using namespace quda;

//- Forward declarations of utility kernels

template <typename T>
__global__ void createGammaGenerators_kernel(Arg_Gamma<T> *arg);

#endif // _UTILITY_KERNELS_H
