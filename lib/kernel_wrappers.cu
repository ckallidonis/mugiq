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
  cpuParam.fieldOrder = unitGamma[0]->FieldOrder();
  cpuParam.siteOrder  = unitGamma[0]->SiteOrder();
  cpuParam.setPrecision(unitGamma[0]->Precision());
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.fieldOrder = unitGamma[0]->FieldOrder();
  cudaParam.siteOrder  = unitGamma[0]->SiteOrder();
  cudaParam.location   = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create     = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(unitGamma[0]->Precision());
  for(int n=0;n<nUnit;n++) gammaGens[n] = ColorSpinorField::Create(cudaParam);

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
  printfQuda("%s: CUDA kernel finished successfully\n", __func__);
  
  //-Create one temporary fine and N-coarse temporary coarse fields
  int nCoarseLevels = mg_env->mgParams->n_level-1;
  int nextCoarse = nCoarseLevels - 1;
  printfQuda("%s: nCoarseLevels = %d , nextCoarse = %d\n", __func__, nCoarseLevels, nextCoarse);
  
  std::vector<ColorSpinorField *> tmpCSF;
  ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
  QudaPrecision coarsePrec = unitGamma[0]->Precision();
  //  ColorSpinorParam csParam(*(gammaGens[0]));
  //QudaPrecision coarsePrec = gammaGens[0]->Precision();
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(coarsePrec);
  
  tmpCSF.push_back(ColorSpinorField::Create(csParam)); //- tmpCSF[0] is a fine field
  printfQuda("%s: mg->B     precision = %d\n", __func__, mg_env->mg_solver->B[0]->Precision());
  printfQuda("%s: gammaGens precision = %d\n", __func__, gammaGens[0]->Precision());
  printfQuda("%s: unitGamma precision = %d\n", __func__, unitGamma[0]->Precision());
  printfQuda("%s: tmpCSF[0] precision = %d\n", __func__, tmpCSF[0]->Precision());
  for(int lev=0;lev<nextCoarse;lev++){
    printfQuda("%s: Got into Create coarse loop lev = %d!!!\n", __func__, lev);
    tmpCSF.push_back(tmpCSF[lev]->CreateCoarse(mg_env->mgParams->geo_block_size[lev],
					       mg_env->mgParams->spin_block_size[lev],
					       mg_env->mgParams->n_vec[lev],
					       coarsePrec,
					       mg_env->mgParams->setup_location[lev+1]));
    printfQuda("%s: tmpCSF[%d] precision = %d\n", __func__, lev+1, tmpCSF[lev+1]->Precision());
  }//-lev
  printfQuda("%s: Temporary fields created successfully\n", __func__);
  
  
  //- Restrict the gamma generators consecutively to get
  //- the unity Gamma vectors at the coarsest level
  for (int n=0; n<nUnit;n++){
    *(tmpCSF[0]) = *(gammaGens[n]);
    for(int lev=0;lev<nextCoarse;lev++){
      printfQuda("%s: Got into Transfer loop lev = %d\n", __func__, lev);
      blas::zero(*tmpCSF[lev+1]);
      if(!mg_env->transfer) errorQuda("%s: For - Transfer operator for level %d does not exist!\n", __func__, lev);
      mg_env->transfer->R(*(tmpCSF[lev+1]), *(tmpCSF[lev]));
      printfQuda("%s: Transfer done in loop for lev = %d\n", __func__, lev);
    }
    blas::zero(*unitGamma[n]);
    if(!mg_env->transferCoarse) errorQuda("%s: Out - Transfer operator for coarsest level does not exist!\n", __func__, nextCoarse);
    //    mg_env->transferLevel[0]->R(*(unitGamma[n]), *(tmpCSF[nextCoarse]));
    mg_env->transferCoarse->R(*(unitGamma[n]), *(tmpCSF[nextCoarse]));
  }

  
  //- Clean-up
  int nTmp = static_cast<int>(tmpCSF.size());
  for(int i=0;i<nTmp;i++)  delete tmpCSF[i];
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

