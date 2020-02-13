#include <loop_coarse_ulocal.h>
#include <kernels_mugiq.h>
#include <tune_quda.h>

//- Shared memory buffer
template <typename Float>
extern __shared__ complex<Float> shMemBuf[];


// //- Wrapper to return the gamma coefficients in constant memory (keep to float for now)
template <typename Float>
inline __device__ const complex<Float>** gammaCoeff() {
  return reinterpret_cast<const complex<Float>**>(gCoeff_cMem);
}

template <typename Float, typename Arg, typename V>
inline __device__ complex<Float> performDotCoarse(V *v1, V *v2, Arg *arg){
  
  complex<Float> r = 1.0;
  return r;
  
//   //- Get the gamma matrix coefficients from constant memory

// #pragma unroll
//   for(int is=0;is<N_SPIN_;is++){     //- Which of the
// #pragma unroll
//     for(int ic=0;ic<N_COLOR_;ic++){  //- generator vectors (color-inside-spin)
// #pragma unroll
//       for(int js=0;js<N_SPIN_;js++){       //- Spin-color index
// #pragma unroll
//         for(int jc=0;jc<N_COLOR_;jc++){    //- within the vector (color-inside-spin)


// 	}// jc
//       }// js
//     }// ic
//   }// is
	  
}

template <typename Float, typename Arg>
__global__ void loop_ulocal_kernel(complex<Float> *loop_dev, Arg *arg){

  typedef typename FieldMapper<Float>::Vector Vector;
  
  int x_cb = blockIdx.x*blockDim.x + threadIdx.x;    // CB site within 4d local volume
  int pty  = blockIdx.y*blockDim.y + threadIdx.y;    // parity within 4d local volume

  int tid  = x_cb + pty * arg->volumeCB;             /* space-time index within result buffer */
  int lV   = arg->volume;

  if (x_cb >= arg->volumeCB) return;
  if (pty  >= arg->nParity) return;
  if (tid  >= lV) return;
  
  const int nG = threadIdx.z; // z-dimension runs on Gamma matrices
  //  int i1 = nG / N_SPIN_;
  //  int i2 = nG % N_SPIN_;


  //- Shared memory storage for coarse eigenvector, coarse gamma unity vector and global result
  //- u is a coarse eigenvector
  //- r is an unphased coarse gamma unity vector
  //- v is a    phased coarse gamma unity vector
  //- globRes is the global (summed result)
  const int coarseSiteLen = arg->Ns * arg->Nc;
  int isite_blk = threadIdx.y * blockDim.x + threadIdx.x;
  complex<Float> *globRes = (complex<Float>*)&(shMemBuf<Float>[arg->shMemElemSite * isite_blk]);
  complex<Float> *dotNoPh = globRes + N_GAMMA_;
  complex<Float> *dotwPh  = globRes + SPINOR_SITE_LEN_;  
  complex<Float> *u = globRes + SPINOR_SITE_LEN_;
  complex<Float> *r = u + coarseSiteLen;
  complex<Float> *v = r + coarseSiteLen;

  const complex<Float> **gCoeff = gammaCoeff<Float>();  

  
  globRes[nG] = 0.0;

  //- Loop over the eigenvectors
#pragma unroll
  for (int m=0; m<arg->nEvec; m++){
    if (0 == nG){
      *(reinterpret_cast<Vector*>(u)) = arg->u[m](x_cb, pty);
    }
    __syncthreads();
    
#pragma unroll
    for(int i=0;i<SPINOR_SITE_LEN_;i++){
      if (0 == nG){
	*(reinterpret_cast<Vector*>(r)) = arg->r[i](x_cb, pty);
	*(reinterpret_cast<Vector*>(v)) = arg->v[i](x_cb, pty);
      }
      __syncthreads();
      
      dotNoPh[i] = performDotCoarse<Float, Arg, Vector>(reinterpret_cast<Vector*>(u), reinterpret_cast<Vector*>(r), arg);
      dotwPh[i]  = performDotCoarse<Float, Arg, Vector>(reinterpret_cast<Vector*>(v), reinterpret_cast<Vector*>(u), arg);
    }
    __syncthreads();
    
#pragma unroll
    for(int s1=0;s1<N_SPIN_;s1++){    //- row index
#pragma unroll
      for(int s2=0;s2<N_SPIN_;s2++){  //- col index
#pragma unroll
        for(int c1=0;c1<N_COLOR_;c1++){
#pragma unroll
          for(int c2=0;c2<N_COLOR_;c2++){
	    int cIdx  = GAMMA_COEFF_IDX(s1,c1,s2,c2);
	    int gIdx1 = GAMMA_GEN_IDX(s1,c1);
	    int gIdx2 = GAMMA_GEN_IDX(s2,c2);

	    globRes[nG] += gCoeff[nG][cIdx] * dotNoPh[gIdx1] * dotwPh[gIdx2];
	  }
	}
      }
    }
    __syncthreads();
    
    /* compute v2^dag . v1 */
//     {
//       complex<float> s = 0;

//       for (int kc = 0 ; kc < QC_Nc ; kc++)
// 	s += conj(v2[QC_LIDX_D_TR(kc, i2)]) * v1[QC_LIDX_D_TR(kc,i1)];

//       globRes[nG] += s;
//     }
//     __syncthreads(); /* sic! avoid overwrite v2 in the next iter; sync globRes before Gamma proj */
  }//- eVecs

//   /* proj on Gamma(nG) -> Corr_dev[tid + lV*nG] */
//   int nGamma  = nG;
//   QC_CPLX s = 0;
// #pragma unroll
//   for (int is = 0 ; is < QC_Ns ; is++) {
//     int js = gamma->left_ind[nGamma][is];
//     s += gamma->left_coeff[nGamma][is] * globRes[QC_LIDX_G(js, is)];
//   }
//   Corr_dev[tid + lV*nGamma] = s;

}//-kernel
//------------------------------------------------------------------------------------------





//-- QUDA/CUDA-tunable Class for creating the ultra-local coarse loop
template <typename Float, int Nspin, int Ncolor, typename Arg>
class assembleLoop_uLocal : public TunableVectorY {
  
protected:
  Arg *arg_dev;
  const ColorSpinorField *meta;
  complex<Float> *loop_dev;
  int Ns;
  int Nc;
  int blockdimZ;
  size_t shmem_per_site;

  //- FIXME
  long long flops() const{
    long long flopCnt = 0;
    flopCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (Nc*Nc*Ns*Ns*Ns * 8 + Ns*Ns*Ns * 2);
    return flopCnt;
  }

  //- FIXME
  long long bytes() const{
    long long byteCnt = 0;
    byteCnt = (long long)meta->VolumeCB() * meta->SiteSubset() * (2*Ns*Ns*Nc*Nc) * 2*8;
    return byteCnt;
  }
  
  bool tuneGridDim() const { return false; }
  unsigned int minThreads() const { return meta->VolumeCB(); }
  
  virtual unsigned int sharedBytesPerBlock(const TuneParam &param) const {
    return param.block.x * param.block.y * shmem_per_site ;
  }

  //- FIXME
  virtual int blockStep() const {
    return 4*((deviceProp.warpSize + blockdimZ - 1) / blockdimZ) ;
  }

  //- FIXME
  virtual int blockMin() const {
    return 4*((deviceProp.warpSize + blockdimZ - 1) / blockdimZ) ;
  }
  
  
public:
  assembleLoop_uLocal(const ColorSpinorField *meta_, Arg *arg_dev_,
		      complex<Float> *loop_dev_,
    		      int blockdimZ_, size_t shmem_per_site_)
    : TunableVectorY(meta_->SiteSubset()), meta(meta_),
      arg_dev(arg_dev_), loop_dev(loop_dev_),
      Ns(Nspin), Nc(Ncolor),
      blockdimZ(blockdimZ_), shmem_per_site(shmem_per_site_)
  {
    strcpy(aux, meta_->AuxString());
    strcat(aux, comm_dim_partitioned_string());

    if(typeid(Float) != typeid(float)) errorQuda("assembleLoop_uLocal: Supports only single-precision for now\n");
  }
  
  virtual ~assembleLoop_uLocal() { }
  
  long long getFlops(){return flops();}
  long long getBytes(){return bytes();}
  
  void apply(const cudaStream_t &stream) {
    TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
    
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("assembleLoop_uLocal::apply(): grid={%ld,%ld,%ld} block={%ld,%ld,%ld} shmem=%ld\n",
		 (long)tp.grid.x, (long)tp.grid.y, (long)tp.grid.z,
		 (long)tp.block.x, (long)tp.block.y, (long)tp.block.z,
		 (long)tp.shared_bytes);
    
    loop_ulocal_kernel<Float, Arg><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(loop_dev, arg_dev);
  }
  
  void initTuneParam(TuneParam &param) const
  {
    TunableVectorY::initTuneParam(param);
    param.block.z = blockdimZ;
    param.grid.z  = 1;
  }
  void defaultTuneParam(TuneParam &param) const
  {
    TunableVectorY::defaultTuneParam(param);
    param.block.z = blockdimZ;
    param.grid.z  = 1;
  }
  
  TuneKey tuneKey() const { return TuneKey(meta->VolString(), typeid(*this).name(), aux); }
};

//- Template on the Precision, Number of Spins and Number of Colors
template <typename Float, int Nspin, int Ncolor>
void assembleCoarseLoop_uLocal(complex<Float> *loop_dev,
			       Eigsolve_Mugiq *eigsolve,
			       const std::vector<ColorSpinorField*> &unitGammaPos,
			       const std::vector<ColorSpinorField*> &unitGammaMom,
			       MugiqLoopParam *loopParams){

  /* Shared memory per site. This should hold:
   * - 1 Nspin*Ncolor ColorSpinor (vector) to store a coarse eigenvector
   * - 1 Nspin*Ncolor ColorSpinor (vector) to store an unphased coarse gamma unity vector
   * - 1 Nspin*Ncolor ColorSpinor (vector) to store a    phased coarse gamma unity vector
   * - 1 Result of Dot product for Nspin*Color generators (unphased) 
   * - 1 Result of Dot product for Nspin*Color generators (phased) 
   * - 16 gamma matrices
   */
  long long shMem_elem_site = 3*Nspin*Ncolor + 2*SPINOR_SITE_LEN_ + N_GAMMA_;
  size_t shMem_bytes_site = sizeof(complex<Float>)*shMem_elem_site;

  //- Create the kernel's argument structure based on template values
  typedef Arg_CoarseLoop_uLocal<Float, Nspin, Ncolor> Arg;
  Arg arg(unitGammaPos, unitGammaMom, eigsolve->getEvecs(), eigsolve->getEvals(), shMem_elem_site);
  Arg *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(Arg));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(Arg), cudaMemcpyHostToDevice);
  checkCudaError();

  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);

  int block_dim_z = N_GAMMA_; //- Number of threads in z-block
  
  //- Create object of loop-assemble class
  assembleLoop_uLocal<Float, Nspin, Ncolor, Arg> loop_uLocal(unitGammaPos[0], arg_dev, loop_dev,
							     block_dim_z, shMem_bytes_site);
  
  if(getVerbosity() >= QUDA_VERBOSE){
    printfQuda("%s: loop_uLocal::Flops = %lld\n", __func__, loop_uLocal.getFlops());
    printfQuda("%s: loop_uLocal::Bytes = %lld\n", __func__, loop_uLocal.getBytes());
  }

  //- Run the CUDA kernel
  double t1 = MPI_Wtime();
  loop_uLocal.apply(0);
  cudaDeviceSynchronize();
  checkCudaError();
  double t2 = MPI_Wtime();
  if(getVerbosity() >= QUDA_VERBOSE)
    printfQuda("TIMING - %s: Ultra-local loop assemble kernel: %f sec\n", __func__, t2-t1);  
  //----------------------------------------------------------

  
  //- Clean-up
  cudaFree(arg_dev);
  arg_dev = nullptr;

  printfQuda("%s: Ultra-local disconnected loop assembly completed\n", __func__);
}
//-------------------------------------------------------------------------------


//- Template on the Precision and Number of Spins
template <typename Float, int Nspin>
void assembleCoarseLoop_uLocal(complex<Float> *loop_dev,
			       Eigsolve_Mugiq *eigsolve,
			       const std::vector<ColorSpinorField*> &unitGammaPos,
			       const std::vector<ColorSpinorField*> &unitGammaMom,
			       MugiqLoopParam *loopParams){

  //- Some sanity checks
  const int nEvec = eigsolve->getEvecs().size();
  const int nColorEv  = eigsolve->getEvecs()[0]->Ncolor();
  
  const int nUnit = unitGammaPos.size();
  const int nColorUnit  = unitGammaPos[0]->Ncolor();

  if(nColorEv != nColorUnit)
    errorQuda("%s: Number of colors between coarse eigenvectors and coarse gamma unity vectors does not match!\n", __func__);

  //- Template on the spins
  if(nColorEv == 3)
    assembleCoarseLoop_uLocal<Float,Nspin,3>(loop_dev, eigsolve, unitGammaPos, unitGammaMom, loopParams);
  else if(nColorEv == 24)
    assembleCoarseLoop_uLocal<Float,Nspin,24>(loop_dev, eigsolve, unitGammaPos, unitGammaMom, loopParams);
  else if(nColorEv == 32)
    assembleCoarseLoop_uLocal<Float,Nspin,32>(loop_dev, eigsolve, unitGammaPos, unitGammaMom, loopParams);
  else
    errorQuda("%s: Unsupported number of colors %d\n", __func__, nColorEv);
}
//-------------------------------------------------------------------------------


/* Top-level function, called from the interface, templated on the Precision
 * This function calls a CUDA kernel to assmeble the disconnected quark loop for ultra-local currents (no displacements).
 * Components are the coarse gamma-matrix unity vectors e_i, the coarse eigenpairs v_i, lambda_i and the gamma matrix coefficients,
 * gcoeff, which already exist in __constant__ memory.
 * The operation performed is:
 * L_n(p,t) = \sum_{m}^{Nev} lambda_m^{-1} u_m^\dag(x) [\sum_{i,j}^{Ns*Nc} gcoeff(n)_{ij} r_i(x) v_j^\dag(y;p,t)] u_m(y)
 */

//- Explicit template instantiations required
template void assembleCoarseLoop_uLocal<double>(complex<double> *loop_dev,
						MG_Mugiq *mg_env, Eigsolve_Mugiq *eigsolve,
						const std::vector<ColorSpinorField*> &unitGammaPos,
						const std::vector<ColorSpinorField*> &unitGammaMom,
						QudaInvertParam *invParams, MugiqLoopParam *loopParams);

template void assembleCoarseLoop_uLocal<float>(complex<float> *loop_dev,
					       MG_Mugiq *mg_env, Eigsolve_Mugiq *eigsolve,
					       const std::vector<ColorSpinorField*> &unitGammaPos,
					       const std::vector<ColorSpinorField*> &unitGammaMom,
					       QudaInvertParam *invParams, MugiqLoopParam *loopParams);

template <typename Float>
void assembleCoarseLoop_uLocal(complex<Float> *loop_dev,
			       MG_Mugiq *mg_env, Eigsolve_Mugiq *eigsolve,
			       const std::vector<ColorSpinorField*> &unitGammaPos,
			       const std::vector<ColorSpinorField*> &unitGammaMom,
			       QudaInvertParam *invParams, MugiqLoopParam *loopParams){
  
  //- Some sanity checks
  const int nSpinEv = eigsolve->getEvecs()[0]->Nspin();
  const QudaPrecision precEv = eigsolve->getEvecs()[0]->Precision(); 
  
  const int nSpinUnit = unitGammaPos[0]->Nspin();
  const QudaPrecision precUnit = unitGammaPos[0]->Precision(); 
  
  if(nSpinEv != nSpinUnit)
    errorQuda("%s: Number of spins between coarse eigenvectors and coarse gamma unity vectors does not match!\n", __func__);
  if(precEv != precUnit)
    errorQuda("%s: Precision between coarse eigenvectors and coarse gamma unity vectors does not match!\n", __func__);

  //- Template on the spins
  if(nSpinEv == 2)
    assembleCoarseLoop_uLocal<Float,2>(loop_dev, eigsolve, unitGammaPos, unitGammaMom, loopParams);
  else
    errorQuda("%s: Unsupported number of spins %d\n", __func__, nSpinEv);
  
}
//-------------------------------------------------------------------------------
