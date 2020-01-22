#include <mg_mugiq.h>
#include <kernel_util.h>
//#include <util_kernels.cuh>

//- Explicit instantiations
template void createGammaCoarseVectors_uLocal<double>(std::vector<ColorSpinorField*> &unitGamma,
						      MG_Mugiq *mg_env, QudaInvertParam *invParams);
template void createGammaCoarseVectors_uLocal<float>(std::vector<ColorSpinorField*> &unitGamma,
						     MG_Mugiq *mg_env, QudaInvertParam *invParams);

template <typename T>
void createGammaCoarseVectors_uLocal(std::vector<ColorSpinorField*> &unitGamma,
				     MG_Mugiq *mg_env, QudaInvertParam *invParams){

  int nUnit = unitGamma.size();
  const int *X = mg_env->mg_solver->B[0]->X();

  std::vector<ColorSpinorField*> gammaGens; // These are fine fields
  gammaGens.resize(nUnit);
  ColorSpinorParam cpuParam(NULL, *invParams, X,
			    invParams->solution_type, invParams->input_location);
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int n=0;n<nUnit;n++)
    gammaGens[n] = ColorSpinorField::Create(cudaParam);

  Arg_Gamma<T>  arg(gammaGens);
  Arg_Gamma<T> *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(Arg_Gamma<T>));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(Arg_Gamma<T>), cudaMemcpyHostToDevice);
  checkCudaError();
  
  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);
  
  //- Call CUDA kernel
  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);  
  createGammaGenerators_kernel<T> <<<gridDim,blockDim>>>(arg_dev);
  cudaDeviceSynchronize();
  checkCudaError();
  

  //-Use transfer->restrictor to obtain unitGamma coarse vectors

  
  //- Clean-up
  for(int n=0;n<nUnit;n++) delete gammaGens[n];
  cudaFree(arg_dev);
  arg_dev = NULL;

  printfQuda("%s: Coarse Gamma Vectors created\n", __func__);
}




#if 0  
  ColorSpinorField *gammaGens[nUnit];
  ColorSpinorParam cpuParam(NULL, *invParams, X,
			    invParams->solution_type, invParams->input_location);
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int n=0;n<nUnit;n++)
    gammaGens[n] = new cudaColorSpinorField(cudaParam);
#endif

