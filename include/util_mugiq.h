#ifndef _MUGIQ_UTIL_H
#define _MUGIQ_UTIL_H

#include <util_quda.h>
#include <enum_mugiq.h>
#include <complex_quda.h>

#define PI 2.0*asin(1.0)

#define MOM_DIM_ 3
#define N_DIM_ 4
#define T_AXIS_ 3
#define N_SPIN_ 4
#define N_COLOR_ 3
#define N_GAMMA_ (N_SPIN_ * N_SPIN_)
#define SPINOR_SITE_LEN_ (N_SPIN_ * N_COLOR_)
#define GAUGE_SITE_LEN_ (N_COLOR_ * N_COLOR_)
#define GAMMA_MAT_ELEM_ (N_SPIN_ * N_SPIN_)

#define SPINOR_SITE_IDX(s,c)  ( (c) + N_COLOR_*(s) )
#define GAUGE_SITE_IDX(c1,c2)  ( (c2) + N_COLOR_*(c1) )

#define GAMMA_MAT_IDX(s1,s2)  ( (s2) + N_SPIN_*(s1) )

#define MOM_MATRIX_IDX(id,im) ( (id) + MOM_DIM_*(im))

#define SHMEM_BLOCK_Z_SIZE (N_GAMMA_)
#define NELEM_SHMEM_CPLX_BUF (2*SPINOR_SITE_LEN_ + N_GAMMA_)

#define THREADS_PER_BLOCK 32

/*
#define checkErrorCudaNoSync() do {                      \
    cudaError_t error = cudaGetLastError();	         \
    if (error != cudaSuccess)			         \
      fprintf(stderr,"(CUDA) %s", cudaGetErrorString(error));	\
  } while (0)


#define checkErrorCuda() do {  \
    cudaDeviceSynchronize();   \
    checkErrorCudaNoSync();    \
  } while (0)
*/

#endif // _MUGIQ_UTIL_H
