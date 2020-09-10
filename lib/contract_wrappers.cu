#include <mugiq_util_kernels.cuh>

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
void performLoopContraction(complex<Float> *loopData_d, ColorSpinorField *evecL, ColorSpinorField *evecR){



}


template void performLoopContraction<float> (complex<float>  *loopData_d, ColorSpinorField *evecL, ColorSpinorField *evecR);
template void performLoopContraction<double>(complex<double> *loopData_d, ColorSpinorField *evecL, ColorSpinorField *evecR);
