#include <mg_mugiq.h>
#include <kernel_util.h>

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
  
  //-Create one temporary fine and N-coarse temporary coarse fields
  int nextCoarse = mg_env->nCoarseLevels - 1;
  
  std::vector<ColorSpinorField *> tmpCSF;
  ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
  QudaPrecision coarsePrec = unitGamma[0]->Precision();
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
  
  //- Restrict the gamma generators consecutively to get
  //- the unity Gamma vectors at the coarsest level
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
  
  //- Clean-up
  int nTmp = static_cast<int>(tmpCSF.size());
  for(int i=0;i<nTmp;i++)  delete tmpCSF[i];
  for(int n=0;n<nUnit;n++) delete gammaGens[n];
  cudaFree(arg_dev);
  arg_dev = NULL;

  printfQuda("%s: Coarse Gamma Vectors created\n", __func__);
}
//-------------------------------------------------------------------------------

/**
 * Create hard-coded gamma coefficients for the DeGrand-Rossi basis
 * Any Gamma matrix can be obtained as G(n) = \sum_{i,j=0}^{3} [c_n(i,j) * e(i) * e^dag(j)],
 * where c(i,j) are the coefficients, and e(i) are the gamma-matrix unity vectors/generators
 */
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
	    int i = GAMMA_GEN_IDX(s1,c1);
	    int j = GAMMA_GEN_IDX(s2,c2);
	    int gIdx = j + SPINOR_SITE_LEN_ * i;

	    if(s2 == column_index[n][s1]) gCoeff[n][gIdx] = {row_value[n][s1][0], row_value[n][s1][1]};
	  }}}}
  }//- n_gamma
  
  
  //- Copy the gamma coeffcients to GPU constant memory
  cudaMemcpyToSymbol(gCoeff_cMem, &gCoeff, sizeof(gCoeff));

  printfQuda("%s: Gamma coefficients created and copied to __constant__ memory\n", __func__);
}


//- Top-level function, called from the interface
void assembleLoopCoarsePart(Eigsolve_Mugiq *eigsolve){
  
  QudaEigParam *eigParams = eigsolve->getEigParams();
  QudaInvertParam *invParams = eigsolve->getInvParams();
  MG_Mugiq *mg_env = eigsolve->getMGEnv();
  
  //- Create coarse gamma-matrix unit vectors
  int nUnit = SPINOR_SITE_LEN_;
  std::vector<ColorSpinorField*> unitGamma; // These are coarse fields

  QudaPrecision ePrec = eigsolve->getEvecs()[0]->Precision();
  if((ePrec != QUDA_DOUBLE_PRECISION) && (ePrec != QUDA_SINGLE_PRECISION))
    errorQuda("%s: Unsupported precision for creating Coarse part of loop\n", __func__);
  else printfQuda("%s: Working in %s precision\n", ePrec == QUDA_DOUBLE_PRECISION ? "double" : "single");
  
  ColorSpinorParam ucsParam(*(eigsolve->getEvecs()[0]));
  ucsParam.create = QUDA_ZERO_FIELD_CREATE;
  ucsParam.location = QUDA_CUDA_FIELD_LOCATION;
  ucsParam.setPrecision(ePrec);
  for(int n=0;n<nUnit;n++)
    unitGamma.push_back(ColorSpinorField::Create(ucsParam));

  if(ePrec == QUDA_DOUBLE_PRECISION)
    createGammaCoarseVectors_uLocal<double>(unitGamma, mg_env, invParams);
  else if(ePrec == QUDA_SINGLE_PRECISION)
    createGammaCoarseVectors_uLocal<float>(unitGamma, mg_env, invParams);
  //-----------------------------------------------------------

  
  //- Create the coefficients of the gamma matrices and copy them to __constant__ memory
  if((invParams->gamma_basis != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) &&
     (mg_env->mgParams->invert_param->gamma_basis != QUDA_DEGRAND_ROSSI_GAMMA_BASIS))
    errorQuda("%s: Supports only DeGrand-Rossi gamma basis\n", __func__);
  if(ePrec == QUDA_DOUBLE_PRECISION) createGammaCoeff<double>();
  else if(ePrec == QUDA_SINGLE_PRECISION) createGammaCoeff<float>(); 
  //-----------------------------------------------------------

  
  //- Clean-up
  for(int n=0;n<nUnit;n++) delete unitGamma[n];

}
//-------------------------------------------------------------------------------
