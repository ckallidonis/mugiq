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


/** Perform contraction/trace:
 * loopData(x) = 1/(sigma) * Tr[ vL(x)^\dag Gamma vR(x) ] 
 *             = 1/(sigma) * \sum_{be,al}{c} conj[vL(x)]_be^c Gamma_{be,al} vR(x)_al^c, where:
 * al, be are spin indices
 * c is a color index
 *
 * Note: The sum over the eignepairs takes place within this kernel as well!
 *
 * We use a 3-dim Block, where:
 * - the x-dimension runs over the checkerboard (even/odd) volume
 * - the y-dimension runs over the parity (even and odd)
 * - the z-dimension runs over the gamma matrices
 * This means that each thread will perform the trace mentioned above for the coordinate space and the "Gamma" space,
 * and will place the result in the corresponding position in the loopData buffer, which has locVol * Ngamma entries.
 *
 * We utilize shared memory, so that the two vectors and the trace result (for each Gamma matrix) reside in a common
 * __shared__ memory buffer, and we define pointers for each object within this buffer.
 *
 */
template <typename Float>
__global__ void loopContract_kernel(complex<Float> *loopData, LoopContractArg<Float> *arg){
  
  //- x-dimension of block working on checkerboard (even/odd) volume
  //- y-dimension of block working on parity (even or odd)
  int x_cb = blockIdx.x*blockDim.x + threadIdx.x;    // checkerboard site within 4d local volume
  int pty  = blockIdx.y*blockDim.y + threadIdx.y;    // parity within 4d local volume
  int tid  = x_cb + pty * arg->volumeCB;             // full site index within result buffer
  int lV   = arg->volume;                            // full local volume

  if (x_cb >= arg->volumeCB) return;
  if (pty  >= arg->nParity) return;
  if (tid  >= lV) return;

  
  //- z-dimension of block working on Gamma matrices
  //- order is: albe = be + N_SPIN_ * al
  //- in fact, it doesn't really matter which of the two indices runs first,
  //- this just determines which z-thread will do  each (al,be) combination
  const int albe = threadIdx.z;
  int al = albe / N_SPIN_ ;
  int be = albe % N_SPIN_ ;

  // Define pointers for the two vectors and the result for each gamma matrix within the shared memory storage
  //- That's the thread index within each block in the x-y plane, and it's common among the z-threads
  int isite_blk = threadIdx.y * blockDim.x + threadIdx.x;
  complex<Float> *vL    = (complex<Float>*)&(shmemBuf<Float>[NELEM_SHMEM_CPLX_BUF*isite_blk]);
  complex<Float> *vR    = vL + SPINOR_SITE_LEN_;
  complex<Float> *resG  = vR + SPINOR_SITE_LEN_;

  //- Get the gamma coefficients from constant memory
  const GammaCoeff<Float> *gamma = gCoeff<Float>();

  //- Get spin-color components of vectors for each lattice site
  typedef typename FieldMapper<Float>::Vector V;
  if(albe == 0) {
    *(reinterpret_cast<V*>(vL)) = arg->eVecL(x_cb, pty);
    *(reinterpret_cast<V*>(vR)) = arg->eVecR(x_cb, pty);
  }
  __syncthreads();
  
  //- trace color indices of vL^dag * vR, resG(be,al) = vL^\dag(be) * vR(a)
  //- In this notation the indices in GAMMA_MAT_IDX(i1, i2) are as:
  //- the first  index corresponds to the left  vector, i1 <-> be
  //- the second index corresponds to the right vector, i2 <-> al
  resG[GAMMA_MAT_IDX(al, be)] = 0.;
  for (int kc=0;kc<N_COLOR_;kc++)
    resG[GAMMA_MAT_IDX(be, al)] += conj(vL[SPINOR_SITE_IDX(be,kc)]) * vR[SPINOR_SITE_IDX(al,kc)];    

  __syncthreads();

  
  //- project/trace on Gamma(iG), trace = resG(be,al) * Gamma(be,al)
  const int iG  = albe;
  complex<Float> trace = 0;
#pragma unroll
  for (int s2=0;s2<N_SPIN_;s2++){
    int s1 = gamma->column_index[iG][s2];
    trace += gamma->row_value[iG][s2] * resG[GAMMA_MAT_IDX(s2, s1)];
  }

  //- Sum over the eigenvalues and scale with the inverse eigenvalue
  loopData[tid + lV*iG] += arg->inv_sigma * trace;    
  
}//- loopContract_kernel


template __global__ void loopContract_kernel<float> (complex<float>  *loopData, LoopContractArg<float>  *arg);
template __global__ void loopContract_kernel<double>(complex<double> *loopData, LoopContractArg<double> *arg);
