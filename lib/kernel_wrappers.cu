#include <mg_mugiq.h>
#include <kernel_util.h>

using namespace quda;


void createGammaCoarseVectors_uLocal(std::vector<ColorSpinorField*> &unitGamma, MG_Mugiq *mg_env){

  int nUnit = unitGamma.size();
  std::vector<ColorSpinorField*> gammaGens; // These are fine fields
  ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int n=0;n<nUnit;n++)
    gammaGens.push_back(ColorSpinorField::Create(csParam));


  //-Create structure passed to the CUDA kernel
  Arg_Gamma<QUDA_DOUBLE_PRECISION>  arg(gammaGens);
  Arg_Gamma<QUDA_DOUBLE_PRECISION> *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(arg));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(arg), cudaMemcpyHostToDevice);
  checkCudaError();
  
  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);
  
#if 0

  //- Call CUDA kernel
  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);  
  createGammaGenerators_kernel<<<gridDim,blockDim>>>(arg_dev);
  cudaDeviceSynchronize();
  checkCudaError();

  cudaFree(arg_dev);
  arg_dev = NULL;

  //-Use transfer->restrictor to obtain unitGamma coarse vectors
  
#endif

  for(int n=0;n<nUnit;n++) delete gammaGens[n];

}
