#include <kernels_mugiq.h>
#include <utility_kernels.h>
#include <gamma.h>

void restrictGammaUnitVectors(std::vector<ColorSpinorField*> &unitGamma,
			      std::vector<ColorSpinorField*> &gammaGens,
			      std::vector<ColorSpinorField*> &tmpCSF,
			      MG_Mugiq *mg_env){

  //- Restrict the gamma generators consecutively to get
  //- the unity Gamma vectors at the coarsest level
  int nUnit = unitGamma.size();
  int nextCoarse = tmpCSF.size() - 1;
  for (int n=0; n<nUnit;n++){
    *(tmpCSF[0]) = *(gammaGens[n]);
    for(int lev=0;lev<nextCoarse;lev++){
      blas::zero(*tmpCSF[lev+1]);
      if(!mg_env->transfer[lev]) errorQuda("%s: For - Transfer operator for level %d does not exist!\n", __func__, lev);
      mg_env->transfer[lev]->R(*(tmpCSF[lev+1]), *(tmpCSF[lev]));
    }
    blas::zero(*unitGamma[n]);
    if(!mg_env->transfer[nextCoarse]) errorQuda("%s: Out - Transfer operator for coarsest level does not exist!\n", __func__, nextCoarse);
    mg_env->transfer[nextCoarse]->R(*(unitGamma[n]), *(tmpCSF[nextCoarse]));
  }
}
//------------------------------------------------------------------------------------


template void createUnphasedGammaUnitVectors<double>(std::vector<ColorSpinorField*> &gammaGens);
template void createUnphasedGammaUnitVectors<float>(std::vector<ColorSpinorField*> &gammaGens);

template <typename Float>
void createUnphasedGammaUnitVectors(std::vector<ColorSpinorField*> &gammaGens){

  //- Create CUDA kernel structure
  ArgGammaPos<Float>  arg(gammaGens);
  ArgGammaPos<Float> *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(ArgGammaPos<Float>));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(ArgGammaPos<Float>), cudaMemcpyHostToDevice);
  checkCudaError();
  
  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);
  
  //- Call CUDA kernel to create Gamma generators in position space
  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
  createGammaGeneratorsPos_kernel<Float> <<<gridDim,blockDim>>>(arg_dev);
  cudaDeviceSynchronize();
  checkCudaError();
  //-----------------------------------------------------------

  //-Clean up
  cudaFree(arg_dev);
  arg_dev = NULL;
}
//------------------------------------------------------------------------------------


template void createPhasedGammaUnitVectors<double>(std::vector<ColorSpinorField*> &gammaGens,
						   std::vector<int> mom, LoopFTSign FTsign);
template void createPhasedGammaUnitVectors<float>(std::vector<ColorSpinorField*> &gammaGens,
						  std::vector<int> mom, LoopFTSign FTsign);

template <typename Float>
void createPhasedGammaUnitVectors(std::vector<ColorSpinorField*> &gammaGens,
				  std::vector<int> mom, LoopFTSign FTsign){

  //- Create CUDA kernel structure
  ArgGammaMom<Float>  arg(gammaGens, mom, FTsign);
  ArgGammaMom<Float> *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(ArgGammaMom<Float>));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(ArgGammaMom<Float>), cudaMemcpyHostToDevice);
  checkCudaError();
  
  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);

  //- Call CUDA kernel to create FT phased Gamma generators
  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, 1);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
  createGammaGeneratorsMom_kernel<Float> <<<gridDim,blockDim>>>(arg_dev);
  cudaDeviceSynchronize();
  checkCudaError();
  //-----------------------------------------------------------

  //-Clean up
  cudaFree(arg_dev);
  arg_dev = NULL;
}
//------------------------------------------------------------------------------------

template void timeDilutePhasedGammaUnitVectors<double>(std::vector<ColorSpinorField*> &gammaGensTD,
						       std::vector<ColorSpinorField*> &gammaGens,
						       int glob_t);
template void timeDilutePhasedGammaUnitVectors<float>(std::vector<ColorSpinorField*> &gammaGensTD,
						      std::vector<ColorSpinorField*> &gammaGens,
						      int glob_t);
template <typename Float>
void timeDilutePhasedGammaUnitVectors<float>(std::vector<ColorSpinorField*> &gammaGensTD,
					     std::vector<ColorSpinorField*> &gammaGens,
					     int glob_t){

  //- Create CUDA Kernel structure
  ArgTimeDilute<Float>  arg(gammaGensTD, gammaGens, glob_t);
  ArgTimeDilute<Float> *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(ArgTimeDilute<Float>));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(ArgTimeDilute<Float>), cudaMemcpyHostToDevice);
  checkCudaError();

  //- Call CUDA kernel to perform time-dilution on FT phased Gamma generators
  //- Here we use a 3-dimensional block-size, 3rd dimension runs over the generators
  int nUnit = gammaGens.size();
  dim3 blockDim(THREADS_PER_BLOCK, arg.nParity, nUnit);
  dim3 gridDim((arg.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);
  timeDilutePhasedGenerators_kernel<Float> <<<gridDim,blockDim>>>(arg_dev);
  cudaDeviceSynchronize();
  checkCudaError();
  
  //-Clean up
  cudaFree(arg_dev);
  arg_dev  = NULL;
}
//------------------------------------------------------------------------------------


/**
 * Create hard-coded gamma coefficients for the DeGrand-Rossi basis
 * Any Gamma matrix can be obtained as G(n) = \sum_{i,j=0}^{3} [c_n(i,j) * e(i) * e^dag(j)],
 * where c(i,j) are the coefficients, and e(i) are the gamma-matrix unity vectors/generators
 */
template void createGammaCoeff<double>(complex<double> gCoeff[][SPINOR_SITE_LEN_*SPINOR_SITE_LEN_]);
template void createGammaCoeff<float>(complex<float> gCoeff[][SPINOR_SITE_LEN_*SPINOR_SITE_LEN_]);

template <typename Float>
void createGammaCoeff(complex<Float> gCoeff[][SPINOR_SITE_LEN_*SPINOR_SITE_LEN_]){

  int nCoeff = N_GAMMA_ * SPINOR_SITE_LEN_ * SPINOR_SITE_LEN_;
  memset(gCoeff, 0, sizeof(complex<Float>)*nCoeff);

  //- The value in rows 0,1,2,3, respectively, of each gamma matrix
  const Float row_value[N_GAMMA_][N_SPIN_][2] = {{ {1,0}, {1,0}, {1,0}, {1,0} },   /* G0 = 1 */
                                                 { {0,1}, {0,1},{0,-1},{0,-1} },   /* G1 = g1 */
                                                 {{-1,0}, {1,0}, {1,0},{-1,0} },   /* G2 = g2 */
                                                 {{0,-1}, {0,1},{0,-1}, {0,1} },   /* G3 = g1 g2 */
                                                 { {0,1},{0,-1},{0,-1}, {0,1} },   /* G4 = g3 */
                                                 {{-1,0}, {1,0},{-1,0}, {1,0} },   /* G5 = g1 g3 */
                                                 {{0,-1},{0,-1},{0,-1},{0,-1} },   /* G6 = g2 g3 */
                                                 { {1,0}, {1,0},{-1,0},{-1,0} },   /* G7 = g1 g2 g3 */
                                                 { {1,0}, {1,0}, {1,0}, {1,0} },   /* G8 = g4 */
                                                 { {0,1}, {0,1},{0,-1},{0,-1} },   /* G9 = g1 g4 */
                                                 {{-1,0}, {1,0}, {1,0},{-1,0} },   /* G10= g2 g4 */
                                                 {{0,-1}, {0,1},{0,-1}, {0,1} },   /* G11= g1 g2 g4 */
                                                 { {0,1},{0,-1},{0,-1}, {0,1} },   /* G12= g3 g4 */
                                                 {{-1,0}, {1,0},{-1,0}, {1,0} },   /* G13= g1 g3 g4 */
                                                 {{0,-1},{0,-1},{0,-1},{0,-1} },   /* G14= g2 g3 g4 */
                                                 { {1,0}, {1,0},{-1,0},{-1,0} }};  /* G15= g1 g2 g3 g4 */

  //- The column in which row_value exists for each gamma matrix
  const int column_index[N_GAMMA_][N_SPIN_] = {{ 0, 1, 2, 3 },   /* G0 = 1 */
                                               { 3, 2, 1, 0 },   /* G1 = g1 */
                                               { 3, 2, 1, 0 },   /* G2 = g2 */
                                               { 0, 1, 2, 3 },   /* G3 = g1 g2 */
                                               { 2, 3, 0, 1 },   /* G4 = g3 */
                                               { 1, 0, 3, 2 },   /* G5 = g1 g3 */
                                               { 1, 0, 3, 2 },   /* G6 = g2 g3 */
                                               { 2, 3, 0, 1 },   /* G7 = g1 g2 g3 */
                                               { 2, 3, 0, 1 },   /* G8 = g4 */
                                               { 1, 0, 3, 2 },   /* G9 = g1 g4 */
                                               { 1, 0, 3, 2 },   /* G10= g2 g4 */
                                               { 2, 3, 0, 1 },   /* G11= g1 g2 g4 */
                                               { 0, 1, 2, 3 },   /* G12= g3 g4 */
                                               { 3, 2, 1, 0 },   /* G13= g1 g3 g4 */
                                               { 3, 2, 1, 0 },   /* G14= g2 g3 g4 */
                                               { 0, 1, 2, 3 }};  /* G15= g1 g2 g3 g4 */

#pragma unroll
  for(int n=0;n<N_GAMMA_;n++){
#pragma unroll
    for(int s1=0;s1<N_SPIN_;s1++){    //- row index
#pragma unroll
      for(int s2=0;s2<N_SPIN_;s2++){  //- col index
#pragma unroll
        for(int c1=0;c1<N_COLOR_;c1++){
#pragma unroll
          for(int c2=0;c2<N_COLOR_;c2++){
            int gIdx = GAMMA_COEFF_IDX(s1,c1,s2,c2); // j + SPINOR_SITE_LEN_ * i;

            if(s2 == column_index[n][s1]) gCoeff[n][gIdx] = {row_value[n][s1][0], row_value[n][s1][1]};
          }}}}
  }//- n_gamma


  //- Copy the gamma coeffcients to GPU constant memory
  cudaMemcpyToSymbol(gCoeff_cMem, &gCoeff, sizeof(complex<Float>)*nCoeff);

  printfQuda("%s: Gamma coefficients created and copied to __constant__ memory\n", __func__);
}
//------------------------------------------------------------------------------------
