#ifndef _UTILITY_KERNELS_H
#define _UTILITY_KERNELS_H

#include <kernels_mugiq.h>

using namespace quda;

//- Forward declarations of utility kernels

template <typename T>
__global__ void createGammaGeneratorsPos_kernel(ArgGammaPos<T> *arg);

template <typename T>
__global__ void createGammaGeneratorsMom_kernel(ArgGammaMom<T> *arg);

template <typename Float>
__global__ void timeDilutePhasedGenerators_kernel(ArgTimeDilute<Float> *arg);

#endif // _UTILITY_KERNELS_H
