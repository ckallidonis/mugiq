#include <gamma.h>
#include <mugiq_util_kernels.cuh>
#include <mugiq_contract_kernels.cuh>
#include <mugiq_displace_kernels.cuh>

template <typename Float>
void copyGammaCoeffStructToSymbol(){

  GammaCoeff<Float> gamma_h;

  for(int m=0;m<N_GAMMA_;m++){
    for(int n=0;n<N_SPIN_;n++){
      gamma_h.column_index[m][n] = GammaColumnIndex(m,n);
      gamma_h.row_value[m][n] = {static_cast<Float>(GammaRowValue(m,n,0)), static_cast<Float>(GammaRowValue(m,n,1))};
    }
  }

  copyGammaCoefftoSymbol<Float>(gamma_h);
}

template void copyGammaCoeffStructToSymbol<float>();
template void copyGammaCoeffStructToSymbol<double>();
//----------------------------------------------------------------------------


template <typename Float>
void copyGammaMapStructToSymbol(){

  GammaMap<Float> map_h;
  
  std::vector<int> minusG = minusGamma();
  std::vector<int> idxG   = indexMapGamma();
  
  std::vector<Float> signGamma(N_GAMMA_,static_cast<Float>(1.0));
  for(auto g: minusG) signGamma.at(g) = static_cast<Float>(-1.0);
  
  for(int m=0;m<N_GAMMA_;m++){
    map_h.sign[m]  = signGamma.at(m);
    map_h.index[m] = idxG.at(m);
  }

  copyGammaMaptoSymbol<Float>(map_h);
}

template void copyGammaMapStructToSymbol<float>();
template void copyGammaMapStructToSymbol<double>();
//----------------------------------------------------------------------------


template <typename Float>
void createPhaseMatrixGPU(complex<Float> *phaseMatrix_d, const int* momMatrix_h,
			  long long locV3, int Nmom, int FTSign,
			  const int localL[], const int totalL[]){

  int *momMatrix_d;
  cudaMalloc((void**)&momMatrix_d, sizeof(int)*Nmom*MOM_DIM_);
  cudaMemcpy(momMatrix_d, momMatrix_h, sizeof(int)*Nmom*MOM_DIM_, cudaMemcpyHostToDevice);
  checkCudaError();

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
  arg_d = nullptr;
}

template void createPhaseMatrixGPU<float>(complex<float> *phaseMatrix_d, const int* momMatrix_h,
					  long long locV3, int Nmom, int FTSign,
					  const int localL[], const int totalL[]);
template void createPhaseMatrixGPU<double>(complex<double> *phaseMatrix_d, const int* momMatrix_h,
					   long long locV3, int Nmom, int FTSign,
					   const int localL[], const int totalL[]);
//----------------------------------------------------------------------------


template <typename Float, QudaFieldOrder fieldOrder>
void performLoopContraction(complex<Float> *loopData_d, ColorSpinorField *eVecL, ColorSpinorField *eVecR, Float sigma){

  typedef LoopContractArg<Float,fieldOrder> Arg;
  
  Arg arg(*eVecL, *eVecR, sigma);
  Arg *arg_d;
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
  loopContract_kernel<Float, Arg><<<gridDim,blockDim,shmemByteSize>>>(loopData_d, arg_d);
  cudaDeviceSynchronize();
  checkCudaError();
  
  cudaFree(arg_d);
  arg_d = nullptr;
}

//- This start to become overwhelming, hopefully no other template parameters will be needed
template void performLoopContraction<float,QUDA_FLOAT2_FIELD_ORDER> (complex<float>  *loopData_d,
								     ColorSpinorField *eVecL, ColorSpinorField *eVecR,
								     float sigma);
template void performLoopContraction<float,QUDA_FLOAT4_FIELD_ORDER> (complex<float>  *loopData_d,
								     ColorSpinorField *eVecL, ColorSpinorField *eVecR,
								     float sigma);
template void performLoopContraction<double,QUDA_FLOAT2_FIELD_ORDER>(complex<double> *loopData_d,
								     ColorSpinorField *eVecL, ColorSpinorField *eVecR,
								     double sigma);
template void performLoopContraction<double,QUDA_FLOAT4_FIELD_ORDER>(complex<double> *loopData_d,
								     ColorSpinorField *eVecL, ColorSpinorField *eVecR,
								     double sigma);
//----------------------------------------------------------------------------


template <typename Float>
void convertIdxOrder_mapGamma(complex<Float> *dataPosMP_d, const complex<Float> *dataPos_d,
			      int nData, int nLoop, int nParity, int volumeCB, const int localL[]){

  //-Some checks
  if(nData != nLoop*N_GAMMA_) errorQuda("%s: This function assumes that nData = nLoop * NGamma\n", __func__);

  ConvertIdxArg arg(nData, nLoop, nParity, volumeCB, localL);
  ConvertIdxArg *arg_d;
  cudaMalloc((void**)&(arg_d), sizeof(arg) );
  checkCudaError();
  cudaMemcpy(arg_d, &arg, sizeof(arg), cudaMemcpyHostToDevice);
  checkCudaError();
  
  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, N_GAMMA_);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
  
  convertIdxOrder_mapGamma_kernel<Float><<<gridDim,blockDim>>>(dataPosMP_d, dataPos_d, arg_d);
  cudaDeviceSynchronize();
  checkCudaError();

  cudaFree(arg_d);
  arg_d = nullptr;
}

template void convertIdxOrder_mapGamma<float> (complex<float> *dataPosMP_d, const complex<float> *dataPos_d,
					       int nData, int nLoop, int nParity, int volumeCB, const int localL[]);
template void convertIdxOrder_mapGamma<double>(complex<double> *dataPosMP_d, const complex<double> *dataPos_d,
					       int nData, int nLoop, int nParity, int volumeCB, const int localL[]);
//----------------------------------------------------------------------------


//-Helper function for exchanging ghosts (boundaries)
void exchangeGhostVec(ColorSpinorField *x){
  const int nFace  = 1;
  x->exchangeGhost((QudaParity)(1), nFace, 0); //- first argument is redundant when nParity = 2. nFace MUST be 1 for now.
}

template <typename Float, QudaFieldOrder order>
void performCovariantDisplacementVector(ColorSpinorField *dst, ColorSpinorField *src, cudaGaugeField *gauge,
					DisplaceDir dispDir, DisplaceSign dispSign){
  exchangeGhostVec(src);

  typedef CovDispVecArg<Float,order> DispArg;

  DispArg arg(*dst, *src, *gauge);
  DispArg *arg_d;
  cudaMalloc((void**)&(arg_d), sizeof(arg));
  checkCudaError();
  cudaMemcpy(arg_d, &arg, sizeof(arg), cudaMemcpyHostToDevice);
  checkCudaError();

  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);


  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);

  covariantDisplacementVector_kernel<Float, DispArg><<<gridDim,blockDim>>>(arg_d, dispDir, dispSign);
  cudaDeviceSynchronize();
  checkCudaError();

  cudaFree(arg_d);
  arg_d = nullptr;

}


template void performCovariantDisplacementVector<float,QUDA_FLOAT2_FIELD_ORDER> (ColorSpinorField *dst,
										 ColorSpinorField *src,
										 cudaGaugeField *gauge,
										 DisplaceDir dispDir, DisplaceSign dispSign);
template void performCovariantDisplacementVector<float,QUDA_FLOAT4_FIELD_ORDER> (ColorSpinorField *dst,
										 ColorSpinorField *src,
										 cudaGaugeField *gauge,
										 DisplaceDir dispDir, DisplaceSign dispSign);
template void performCovariantDisplacementVector<double,QUDA_FLOAT2_FIELD_ORDER>(ColorSpinorField *dst,
										 ColorSpinorField *src,
										 cudaGaugeField *gauge,
										 DisplaceDir dispDir, DisplaceSign dispSign);
template void performCovariantDisplacementVector<double,QUDA_FLOAT4_FIELD_ORDER>(ColorSpinorField *dst,
										 ColorSpinorField *src,
										 cudaGaugeField *gauge,
										 DisplaceDir dispDir, DisplaceSign dispSign);
//----------------------------------------------------------------------------
