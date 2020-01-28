#include <loop_coarse.h>
#include <kernels_mugiq.h>
#include <tune_quda.h>

  //-- QUDA/CUDA-tunable Class for creating the ultra-local coarse loop
template <typename Float>
class assembleLoop_uLocal : public TunableVectorY {
  
protected:
  void *arg_dev;
  const cudaColorSpinorField *meta;
  complex<Float> *loop_dev;
  int Nc;
  int Ns;
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
  assembleLoop_uLocal(const cudaColorSpinorField *meta_, void *arg_dev_,
		      complex<Float> *loop_dev_,
		      int blockdimZ_, size_t shmem_per_site_)
    : TunableVectorY(meta_->SiteSubset()), meta(meta_),
      arg_dev(arg_dev_), loop_dev(loop_dev_),
      Nc(N_COLOR_), Ns(N_SPIN_),
      blockdimZ(blockdimZ_), shmem_per_site(shmem_per_site_)
  {
    strcpy(aux, meta_->AuxString());
    strcat(aux, comm_dim_partitioned_string());
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
    
    loop_ulocal_kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(loop_dev, (Arg_Loop_uLocal<Float>*)arg_dev);
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



/* Top-level function, called from the interface
 * This function calls a CUDA kernel to assmeble the disconnected quark loop for ultra-local currents (no displacements).
 * Components are the coarse gamma-matrix unity vectors e_i, the coarse eigenpairs v_i, lambda_i and the gamma matrix coefficients,
 * gcoeff, which are already copied to __constant__ memory.
 * The operation performed is:
 * L_n(x,t) = \sum_{m}^{Nev} lambda_m^{-1} v_m^\dag [\sum_{i,j}^{Ns*Nc} gcoeff(n)_{ij} e_i e_j^\dag] v_m
 */
template void assembleLoopCoarsePart_uLocal<double>(complex<double> *loop_h,
						    Eigsolve_Mugiq *eigsolve,
						    const std::vector<ColorSpinorField*> &unitGamma);
template void assembleLoopCoarsePart_uLocal<float>(complex<float> *loop_h,
						   Eigsolve_Mugiq *eigsolve,
						   const std::vector<ColorSpinorField*> &unitGamma);

template <typename Float>
void assembleLoopCoarsePart_uLocal(complex<Float> *loop_h,
				   Eigsolve_Mugiq *eigsolve,
				   const std::vector<ColorSpinorField*> &unitGamma){

  //- Create the kernel's argument structure
  Arg_Loop_uLocal<Float> arg(unitGamma, eigsolve->getEvecs(), eigsolve->getEvals());
  Arg_Loop_uLocal<Float> *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(Arg_Loop_uLocal<Float>));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(Arg_Loop_uLocal<Float>), cudaMemcpyHostToDevice);
  checkCudaError();

  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);

  //- Allocate the device loop buffer
  if(loop_h == NULL) errorQuda("%s: Host loop buffer not allocated correctly\n", __func__);
  complex<Float> *loop_dev = NULL;
  cudaMalloc((void**)&loop_dev, sizeof(loop_h));
  checkCudaError();
  cudaMemset(loop_dev, 0, sizeof(loop_h));

  //- Create object of loop-assemble class, run the kernel
  //- TODO

  cudaMemcpy(loop_h, loop_dev, sizeof(loop_h), cudaMemcpyDeviceToHost);

  //- Clean-up
  cudaFree(arg_dev);
  arg_dev = nullptr;
  cudaFree(loop_dev);
  loop_dev = nullptr;

  printfQuda("%s: Ultra-local disconnected loop assembly completed\n", __func__);
}
//-------------------------------------------------------------------------------
