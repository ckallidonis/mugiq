#include <mg_mugiq.h>
#include <eigsolve_mugiq.h>
#include <utility_kernels.h>
#include <interface_mugiq.h>

template void createGammaCoarseVectors_uLocal<double>(std::vector<ColorSpinorField*> &unitGammaPos,
						      std::vector<ColorSpinorField*> &unitGammaMom,
						      MG_Mugiq *mg_env, QudaInvertParam *invParams,
						      MugiqLoopParam *loopParams);
template void createGammaCoarseVectors_uLocal<float>(std::vector<ColorSpinorField*> &unitGammaPos,
						     std::vector<ColorSpinorField*> &unitGammaMom,
						     MG_Mugiq *mg_env, QudaInvertParam *invParams,
						     MugiqLoopParam *loopParams);

template <typename Float>
void createGammaCoarseVectors_uLocal(std::vector<ColorSpinorField*> &unitGammaPos,
				     std::vector<ColorSpinorField*> &unitGammaMom,
				     MG_Mugiq *mg_env, QudaInvertParam *invParams,
				     MugiqLoopParam *loopParams){

  int nUnit = unitGammaPos.size();
  const int *X = mg_env->mg_solver->B[0]->X();


  //- Create one temporary fine and N_coarse-1 temporary coarse fields
  //- Will be used for restricting the fine Gamma generators for
  //- both momentum and position space
  int nextCoarse = mg_env->nCoarseLevels - 1;
  
  std::vector<ColorSpinorField *> tmpCSF;
  ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
  QudaPrecision coarsePrec = unitGammaPos[0]->Precision();
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(coarsePrec);
  
  tmpCSF.push_back(ColorSpinorField::Create(csParam)); //- tmpCSF[0] is a fine field
  for(int lev=0;lev<nextCoarse;lev++){
    tmpCSF.push_back(tmpCSF[lev]->CreateCoarse(mg_env->mgParams->geo_block_size[lev],
					       mg_env->mgParams->spin_block_size[lev],
					       mg_env->mgParams->n_vec[lev],
					       coarsePrec,
					       mg_env->mgParams->setup_location[lev+1]));
  }//-lev
  //-----------------------------------------------------------
  
  //- These are fine fields, will be used for both
  //- position and momentum space generators
  std::vector<ColorSpinorField*> gammaGens;
  gammaGens.resize(nUnit);
  ColorSpinorParam cpuParam(NULL, *invParams, X,
			    invParams->solution_type, invParams->input_location);
  cpuParam.fieldOrder = unitGammaPos[0]->FieldOrder();
  cpuParam.siteOrder  = unitGammaPos[0]->SiteOrder();
  cpuParam.setPrecision(unitGammaPos[0]->Precision());
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.fieldOrder = unitGammaPos[0]->FieldOrder();
  cudaParam.siteOrder  = unitGammaPos[0]->SiteOrder();
  cudaParam.location   = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create     = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(unitGammaPos[0]->Precision());
  for(int n=0;n<nUnit;n++)
    gammaGens[n] = ColorSpinorField::Create(cudaParam);
  //-----------------------------------------------------------
  
  //- Create the position-space unity vectors
  ArgGammaPos<Float>  argPos(gammaGens);
  ArgGammaPos<Float> *argPos_dev;
  cudaMalloc((void**)&(argPos_dev), sizeof(ArgGammaPos<Float>));
  checkCudaError();
  cudaMemcpy(argPos_dev, &argPos, sizeof(ArgGammaPos<Float>), cudaMemcpyHostToDevice);
  checkCudaError();
  
  if(argPos.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);
  
  //- Call CUDA kernel to create Gamma generators in position space
  dim3 blockDim(THREADS_PER_BLOCK, argPos.nParity, 1);
  dim3 gridDim((argPos.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);  
  createGammaGeneratorsPos_kernel<Float> <<<gridDim,blockDim>>>(argPos_dev);
  cudaDeviceSynchronize();
  checkCudaError();
  //-----------------------------------------------------------
  
  //- Restrict the gamma generators in position space consecutively to get
  //- the unity Gamma vectors at the coarsest level
  for (int n=0; n<nUnit;n++){
    *(tmpCSF[0]) = *(gammaGens[n]);
    for(int lev=0;lev<nextCoarse;lev++){
      blas::zero(*tmpCSF[lev+1]);
      if(!mg_env->transfer[lev]) errorQuda("%s: For - Transfer operator for level %d does not exist!\n", __func__, lev);
      mg_env->transfer[lev]->R(*(tmpCSF[lev+1]), *(tmpCSF[lev]));
    }
    blas::zero(*unitGammaPos[n]);
    if(!mg_env->transfer[nextCoarse]) errorQuda("%s: Out - Transfer operator for coarsest level does not exist!\n", __func__, nextCoarse);
    mg_env->transfer[nextCoarse]->R(*(unitGammaPos[n]), *(tmpCSF[nextCoarse]));
  }
  printfQuda("%s: Coarse Gamma Vectors in position space created\n", __func__);
  //-----------------------------------------------------------


  //- Create the momentum-space gamma matrix generators
  ArgGammaMom<Float> *argMom_dev;
  cudaMalloc((void**)&(argMom_dev), sizeof(ArgGammaMom<Float>));
  checkCudaError();
  for(int p=0;p<loopParams->Nmom;p++){
    std::vector<int> mom = loopParams->momMatrix[p];
    ArgGammaMom<Float>  argMom(gammaGens, mom, loopParams->FTSign);
    cudaMemcpy(argMom_dev, &argMom, sizeof(ArgGammaMom<Float>), cudaMemcpyHostToDevice);
    checkCudaError();
    
    //- Call CUDA kernel to create Gamma generators in position space
    dim3 blockDim(THREADS_PER_BLOCK, argMom.nParity, 1);
    dim3 gridDim((argMom.volumeCB + blockDim.x -1)/blockDim.x, 1, 1);  
    createGammaGeneratorsMom_kernel<Float> <<<gridDim,blockDim>>>(argMom_dev);
    cudaDeviceSynchronize();
    checkCudaError();
    //-----------------------------------------------------------


    printfQuda("%s: Phased Coarse Gamma Vectors for momentum (%+02d,%+02d,%+02d) created\n", __func__, mom[0], mom[1], mom[2]);
  }//- momentum

  
  
  //- Clean-up
  int nTmp = static_cast<int>(tmpCSF.size());
  for(int i=0;i<nTmp;i++)  delete tmpCSF[i];
  for(int n=0;n<nUnit;n++) delete gammaGens[n];
  cudaFree(argPos_dev);
  cudaFree(argMom_dev);
  argPos_dev = NULL;
  argMom_dev = NULL;
}
//-------------------------------------------------------------------------------


/**
 * Create hard-coded gamma coefficients for the DeGrand-Rossi basis
 * Any Gamma matrix can be obtained as G(n) = \sum_{i,j=0}^{3} [c_n(i,j) * e(i) * e^dag(j)],
 * where c(i,j) are the coefficients, and e(i) are the gamma-matrix unity vectors/generators
 */
template void createGammaCoeff<double>();
template void createGammaCoeff<float>();

template <typename Float> 
void createGammaCoeff(){

  complex<Float> gCoeff[N_GAMMA_][SPINOR_SITE_LEN_*SPINOR_SITE_LEN_];
  memset(gCoeff, 0, sizeof(gCoeff));

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
  cudaMemcpyToSymbol(gCoeff_cMem, &gCoeff, sizeof(gCoeff));

  printfQuda("%s: Gamma coefficients created and copied to __constant__ memory\n", __func__);
}
//-------------------------------------------------------------------------------
