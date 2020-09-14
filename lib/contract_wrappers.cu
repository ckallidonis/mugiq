#include <mugiq_util_kernels.cuh>
#include <mugiq_contract_kernels.cuh>

template <typename Float>
void copyGammaCoeffStructToSymbol(){

  GammaCoeff<Float> gamma_h;

  for(int m=0;m<N_GAMMA_;m++){
    for(int n=0;n<N_SPIN_;n++){
      gamma_h.column_index[m][n] = GammaColumnIndex[m][n];
      gamma_h.row_value[m][n] = {static_cast<Float>(GammaRowValue[m][n][0]), static_cast<Float>(GammaRowValue[m][n][1])};
    }
  }
  
  cudaMemcpyToSymbol(cGamma, &gamma_h, sizeof(GammaCoeff<Float>));
}

template void copyGammaCoeffStructToSymbol<float>();
template void copyGammaCoeffStructToSymbol<double>();
//----------------------------------------------------------------------------


template <typename Float>
void createPhaseMatrixGPU(complex<Float> *phaseMatrix_d, const int* momMatrix_h,
			  long long locV3, int Nmom, int FTSign,
			  const int localL[], const int totalL[]){

  int* momMatrix_d;
  cudaMalloc((void**)&momMatrix_d, sizeof(momMatrix_h));
  checkCudaError();
  cudaMemcpy(momMatrix_d, momMatrix_h, sizeof(momMatrix_h), cudaMemcpyHostToDevice);

  MomProjArg arg(locV3, Nmom, FTSign, localL, totalL);
  MomProjArg *arg_d;
  cudaMalloc((void**)&(arg_d), sizeof(MomProjArg) );
  checkCudaError();
  cudaMemcpy(arg_d, &arg, sizeof(MomProjArg), cudaMemcpyHostToDevice);

  //-Call the kernel
  dim3 blockDim(THREADS_PER_BLOCK, 1, 1);
  dim3 gridDim((locV3 + blockDim.x -1)/blockDim.x, 1, 1); // spawn threads only for the spatial volume

  phaseMatrix_kernel<Float><<<gridDim,blockDim>>>(phaseMatrix_d, momMatrix_d, arg_d);
  cudaDeviceSynchronize();
  checkCudaError();

  cudaFree(momMatrix_d);
  cudaFree(arg_d);
  
}

template void createPhaseMatrixGPU<float>(complex<float> *phaseMatrix_d, const int* momMatrix_h,
					  long long locV3, int Nmom, int FTSign,
					  const int localL[], const int totalL[]);
template void createPhaseMatrixGPU<double>(complex<double> *phaseMatrix_d, const int* momMatrix_h,
					   long long locV3, int Nmom, int FTSign,
					   const int localL[], const int totalL[]);
//----------------------------------------------------------------------------


template <typename Float>
void performLoopContraction(complex<Float> *loopData_d, ColorSpinorField *eVecL, ColorSpinorField *eVecR, Float sigma){

  LoopContractArg<Float> arg(eVecL, eVecR, sigma);
  LoopContractArg<Float> *arg_d;
  cudaMalloc((void**)&(arg_d), sizeof(arg) );
  checkCudaError();
  cudaMemcpy(arg_d, &arg, sizeof(arg), cudaMemcpyHostToDevice);
  checkCudaError();

  if(arg.nParity != 2) errorQuda("%s: Loop contraction kernels support only Full Site Subset spinors!\n", __func__);

  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, SHMEM_BLOCK_Z_SIZE);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);  

  //- Size of the required shared memory in bytes
  size_t shmemByteSize = sizeof(complex<Float>) * NELEM_SHMEM_CPLX_BUF * blockDim.x * blockDim.y;
  
  //-Call the kernel
  loopContract_kernel<Float><<<gridDim,blockDim,shmemByteSize>>>(loopData_d, arg_d);
  cudaDeviceSynchronize();
  checkCudaError();
  
  cudaFree(arg_d);  
}


template void performLoopContraction<float> (complex<float>  *loopData_d,
					     ColorSpinorField *evecL, ColorSpinorField *evecR, float sigma);
template void performLoopContraction<double>(complex<double> *loopData_d,
					     ColorSpinorField *evecL, ColorSpinorField *evecR, double sigma);
//----------------------------------------------------------------------------



template <typename Float>
void convertIdxOrderToMomProj(complex<Float> *dataPosMP_d, const complex<Float> *dataPos_d,
                              int Ndata, int nParity, int volumeCB, const int localL[]){

  ConvertIdxArg arg(Ndata, nParity, volumeCB, localL);
  ConvertIdxArg *arg_d;
  cudaMalloc((void**)&(arg_d), sizeof(arg) );
  checkCudaError();
  cudaMemcpy(arg_d, &arg, sizeof(arg), cudaMemcpyHostToDevice);
  checkCudaError();
  
  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, Ndata);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
  
  convertIdxMomProj_kernel<Float><<<gridDim,blockDim>>>(dataPosMP_d, dataPos_d, arg_d);
  cudaDeviceSynchronize();
  checkCudaError();

  cudaFree(arg_d);
  arg_d = NULL;

}


template void convertIdxOrderToMomProj<float> (complex<float> *dataPosMP_d, const complex<float> *dataPos_d,
					       int Ndata, int nParity, int volumeCB, const int localL[]);
template void convertIdxOrderToMomProj<double>(complex<double> *dataPosMP_d, const complex<double> *dataPos_d,
					       int Ndata, int nParity, int volumeCB, const int localL[]);
//----------------------------------------------------------------------------
