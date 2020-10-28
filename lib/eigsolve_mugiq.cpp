#include <eigsolve_mugiq.h>
#include <util_quda.h>
#include <eigensolve_quda.h>

Eigsolve_Mugiq::Eigsolve_Mugiq(MugiqEigParam *eigParams_,
			       MG_Mugiq *mg_env_,
			       TimeProfile *eigProfile_,
			       MuGiqBool computeCoarse_) :
  eigParams(eigParams_),
  eigInit(MUGIQ_BOOL_FALSE),
  useMGenv(MUGIQ_BOOL_TRUE),
  mg_env(mg_env_),
  mgParams(mg_env->mgParams),
  invParams(eigParams->QudaEigParams->invert_param),
  eigProfile(eigProfile_),
  dirac(nullptr),
  mat(nullptr),
  diracCreated(MUGIQ_BOOL_FALSE),
  eVals_quda(nullptr),
  eVals(nullptr),
  eVals_sigma(nullptr),
  evals_res(nullptr),
  computeCoarse(computeCoarse_)
{
  if(!mg_env->mgInit) errorQuda("%s: Multigrid environment must be initialized before Eigensolver.\n", __func__);
  
  if(computeCoarse){
    allocateCoarseEvecs();

    //- The Coarse Dirac operator
    //- This is diracCoarseResidual of the QUDA MG class
    dirac = mg_env->diracCoarse;
    if(typeid(*dirac) != typeid(DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");
  }
  else{
    allocateFineEvecs();

    //- The fine Dirac operator
    //- This is diracResidual of the QUDA MG class.
    //- Equivalently: dirac = mg_env->mg_solver->mgParam->matResidual.Expose() = mg_env->mg_solver->d
    dirac = mg_env->mg_solver->d;    
  }
  
  //- Determine and create the Dirac matrix whose eigenvalues will be computed
  eigParams->diracType = determineEigOperator(eigParams->QudaEigParams);
  createNewDiracMatrix();

  allocateEvals();
  
  makeChecks();  
  eigInit = MUGIQ_BOOL_TRUE;
}

//- This constructor is used when NO Multi-grid is involved
Eigsolve_Mugiq::Eigsolve_Mugiq(MugiqEigParam *eigParams_,
			       TimeProfile *eigProfile_) :
  eigParams(eigParams_),
  eigInit(MUGIQ_BOOL_FALSE),
  useMGenv(MUGIQ_BOOL_FALSE),
  mg_env(nullptr),
  mgParams(nullptr),
  invParams(eigParams->QudaEigParams->invert_param),
  eigProfile(eigProfile_),
  dirac(nullptr),
  mat(nullptr),
  diracCreated(MUGIQ_BOOL_FALSE),
  eVals_quda(nullptr),
  eVals(nullptr),
  eVals_sigma(nullptr),
  evals_res(nullptr),
  computeCoarse(MUGIQ_BOOL_FALSE)
{  
  allocateFineEvecs();

  createDiracOperator();
  
  //- Determine and create the Dirac matrix whose eigenvalues will be computed
  eigParams->diracType = determineEigOperator(eigParams->QudaEigParams);
  createNewDiracMatrix();

  allocateEvals();

  makeChecks();
  eigInit = MUGIQ_BOOL_TRUE;
}


Eigsolve_Mugiq::~Eigsolve_Mugiq(){  
  for(int i=0;i<eigParams->nEv;i++) delete eVecs[i];
  delete eVals_quda;
  delete eVals;
  delete evals_res;
  if(eigParams->diracType == MUGIQ_EIG_OPERATOR_MdagM || eigParams->diracType == MUGIQ_EIG_OPERATOR_MMdag)
    delete eVals_sigma;

  if(mat) delete mat;
  mat = nullptr;
  
  if(useMGenv){
    //- (dirac deletion is taken care by mg_solver destructor in this case)
    int nTmp = static_cast<int>(tmpCSF.size());
    for(int i=0;i<nTmp;i++) if(tmpCSF[i]) delete tmpCSF[i];    
  }
  
  if(dirac && diracCreated){
    delete dirac;
    dirac = nullptr;
  }
  
  eigInit = MUGIQ_BOOL_FALSE;
}


void Eigsolve_Mugiq::allocateFineEvecs(){
  cudaGaugeField *gauge = checkGauge(invParams);
  const int *X = gauge->X(); //- The Lattice Size
  
  ColorSpinorParam csParam(nullptr, *invParams, X, false, QUDA_CUDA_FIELD_LOCATION); // false is for pc_solution
  csParam.print();
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(eigParams->QudaEigParams->cuda_prec_ritz, eigParams->QudaEigParams->cuda_prec_ritz, true);
  
  for(int i=0;i<eigParams->nEv;i++) eVecs.push_back(ColorSpinorField::Create(csParam));
}


void Eigsolve_Mugiq::allocateCoarseEvecs(){
  //- Use Quda's Null vectors as reference for the fine color-spinor
  ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
  QudaPrecision coarsePrec = invParams->cuda_prec;
  csParam.location = QUDA_CUDA_FIELD_LOCATION;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(coarsePrec);
  
  int nInterLevels = mg_env->nInterLevels;
  
  //-Create coarse fields and allocate coarse eigenvectors recursively
  tmpCSF.push_back(ColorSpinorField::Create(csParam)); //- tmpCSF[0] is a fine field
  for(int lev=0;lev<nInterLevels;lev++){
    tmpCSF.push_back(tmpCSF[lev]->CreateCoarse(mgParams->geo_block_size[lev],
					       mgParams->spin_block_size[lev],
					       mgParams->n_vec[lev],
					       coarsePrec,
					       mgParams->setup_location[lev+1]));
  }//-lev
  
  for(int i=0;i<eigParams->nEv;i++){
    eVecs.push_back(tmpCSF[nInterLevels]->CreateCoarse(mgParams->geo_block_size[nInterLevels],
						       mgParams->spin_block_size[nInterLevels],
						       mgParams->n_vec[nInterLevels],
						       coarsePrec,
						       mgParams->setup_location[nInterLevels+1]));
  }
}


void Eigsolve_Mugiq::allocateEvals(){  
  if(eigParams->diracType == MUGIQ_EIG_OPERATOR_INVALID)
    errorQuda("%s: Dirac Matrix type must be defined first!", __func__);

  eVals_quda  = new std::vector<Complex>(eigParams->nEv, 0.0); // These come from the QUDA eigensolver
  eVals       = new std::vector<Complex>(eigParams->nEv, 0.0); // These are computed from the Eigsolve_Mugiq class
  evals_res   = new std::vector<double>(eigParams->nEv, 0.0);  // The eigenvalues residual
  if(eigParams->diracType == MUGIQ_EIG_OPERATOR_MdagM || eigParams->diracType == MUGIQ_EIG_OPERATOR_MMdag)
    eVals_sigma = new std::vector<double>(eigParams->nEv, 0.0); // Singular values of M, eigenvalues of Hermitian g5*M
}


void Eigsolve_Mugiq::createDiracOperator(){
  //- Whether we are running for the "full" or even-odd preconditioned Operator
  bool pc_solve = (invParams->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (invParams->solve_type == QUDA_NORMOP_PC_SOLVE) ||
    (invParams->solve_type == QUDA_NORMERR_PC_SOLVE);
  
  //- Create the Dirac operator
  DiracParam diracParam;
  setDiracParam(diracParam, invParams, pc_solve);
  dirac = Dirac::create(diracParam);

  diracCreated = MUGIQ_BOOL_TRUE;
}


MuGiqEigOperator Eigsolve_Mugiq::determineEigOperator(QudaEigParam *QudaEigParams_){
  MuGiqEigOperator dType = MUGIQ_EIG_OPERATOR_INVALID;
  
  if (!QudaEigParams_->use_norm_op && !QudaEigParams_->use_dagger)     dType = MUGIQ_EIG_OPERATOR_M;
  else if (!QudaEigParams_->use_norm_op && QudaEigParams_->use_dagger) dType = MUGIQ_EIG_OPERATOR_Mdag;
  else if (QudaEigParams_->use_norm_op && !QudaEigParams_->use_dagger) dType = MUGIQ_EIG_OPERATOR_MdagM;
  else if (QudaEigParams_->use_norm_op && QudaEigParams_->use_dagger)  dType = MUGIQ_EIG_OPERATOR_MMdag;
  else errorQuda("%s: Cannot determine Dirac Operator type from QudaEigParams\n", __func__);

  return dType;
}


void Eigsolve_Mugiq::createNewDiracMatrix(){
  if(!dirac) errorQuda("%s: Dirac operator must be defined first!", __func__);
  if(eigParams->diracType == MUGIQ_EIG_OPERATOR_INVALID)
    errorQuda("%s: Dirac Matrix type must be defined first!", __func__);
  
  if      (eigParams->diracType == MUGIQ_EIG_OPERATOR_M)      mat = new DiracM(*dirac);	  
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_Mdag)   mat = new DiracMdag(*dirac); 
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_MdagM)  mat = new DiracMdagM(*dirac);
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_MMdag)  mat = new DiracMMdag(*dirac);
  else errorQuda("%s: Unsupported Dirac operator type\n", __func__);
}


void Eigsolve_Mugiq::makeChecks(){

  if(!mat) errorQuda("%s: Dirac operator is not defined.\n", __func__);
  
  if (invParams->dslash_type != QUDA_WILSON_DSLASH && invParams->dslash_type != QUDA_CLOVER_WILSON_DSLASH)
    errorQuda("%s: Supports only Wilson and Wilson-Clover operators for now!\n", __func__);

  // No polynomial acceleration on non-symmetric matrices
  if (eigParams->QudaEigParams->use_poly_acc &&
      !eigParams->QudaEigParams->use_norm_op && !(invParams->dslash_type == QUDA_LAPLACE_DSLASH))
    errorQuda("%s: Polynomial acceleration with non-symmetric matrices not supported", __func__);
}


void Eigsolve_Mugiq::printInfo(){

  if(!eigInit) errorQuda("%s: Eigsolve_Mugiq must be initialized first.\n", __func__);

  char optrStr[6];
  switch (eigParams->diracType) {
  case MUGIQ_EIG_OPERATOR_M:     strcpy(optrStr,"M");     break;
  case MUGIQ_EIG_OPERATOR_Mdag:  strcpy(optrStr,"Mdag");  break;
  case MUGIQ_EIG_OPERATOR_MdagM: strcpy(optrStr,"MdagM"); break;
  case MUGIQ_EIG_OPERATOR_MMdag: strcpy(optrStr,"MMdag"); break;
  default: break;
  }

  char eig_algo[7];
  switch (eigParams->QudaEigParams->eig_type) {
  case QUDA_EIG_TR_LANCZOS: strcpy(eig_algo, "TRL");    break;
  case QUDA_EIG_IR_LANCZOS: strcpy(eig_algo, "IRL");    break;
  case QUDA_EIG_IR_ARNOLDI: strcpy(eig_algo, "IRA");    break;
  case QUDA_EIG_PRIMME:     strcpy(eig_algo, "PRIMME"); break;
  default: break;
  }

  char spectrum[3];
  switch (eigParams->QudaEigParams->spectrum) {
  case QUDA_SPECTRUM_SR_EIG: strcpy(spectrum, "SR"); break;
  case QUDA_SPECTRUM_LR_EIG: strcpy(spectrum, "LR"); break;
  case QUDA_SPECTRUM_SM_EIG: strcpy(spectrum, "SM"); break;
  case QUDA_SPECTRUM_LM_EIG: strcpy(spectrum, "LM"); break;
  case QUDA_SPECTRUM_SI_EIG: strcpy(spectrum, "SI"); break;
  case QUDA_SPECTRUM_LI_EIG: strcpy(spectrum, "LI"); break;
  default: break;
  }
    
  printfQuda("\n\n*****************************************\n");
  printfQuda("           Eigensolve MUGIQ INFO\n");
  printfQuda("Will %suse the Multigrid environment\n", useMGenv ? "" : "NOT ");
  if(useMGenv) printfQuda("Will %suse the Coarse Dirac operator\n", computeCoarse ? "" : "NOT ");
  printfQuda("Will compute the eigenpairs of the %s Dirac operator\n", optrStr);
  printfQuda("Will employ the %s algorithm for computation\n", eig_algo);
  printfQuda("Part of spectrum requested: %s\n", spectrum);
  printfQuda("Number of eigenvalues requested: %d\n", eigParams->nEv);
  printfQuda("Size of Krylov space: %d\n", eigParams->nKr);
  printfQuda("Requested tolerance is: %e\n", eigParams->tol);
  printfQuda("Will %suse Chebyshev Polynomial Acceleration\n", eigParams->use_poly_acc ? "" : "NOT ");
  if(eigParams->use_poly_acc){
    printfQuda("  Poly Acc. Degree: %d\n", eigParams->poly_acc_deg);
    printfQuda("  Poly Acc. a_min : %e\n", eigParams->a_min);
    printfQuda("  Poly Acc. a_max : %e\n", eigParams->a_max);
  }  
  printfQuda("*****************************************\n\n");

}


void Eigsolve_Mugiq::computeEvecs(){

  if(!eigInit) errorQuda("%s: Eigsolve_Mugiq must be initialized first.\n", __func__);

  //- Perform eigensolve
  EigenSolver *eigSolve = EigenSolver::create(eigParams->QudaEigParams, *mat, *eigProfile);
  (*eigSolve)(eVecs, *eVals_quda);

  delete eigSolve;
}

void Eigsolve_Mugiq::computeEvals(){

  ColorSpinorParam csParam(*eVecs[0]);
  ColorSpinorField *w;
  w = ColorSpinorField::Create(csParam);

  std::vector<Complex> &lambda = *eVals;
  std::vector<double> &r = *evals_res;

  double kappa = invParams->kappa;
  
  for(int i=0; i<eigParams->nEv; i++){
    (*mat)(*w,*eVecs[i]); //- w = M*v_i
    if(invParams->mass_normalization == QUDA_MASS_NORMALIZATION) blas::ax(0.25/(kappa*kappa), *w);
    lambda[i] = blas::cDotProduct(*eVecs[i], *w) / sqrt(blas::norm2(*eVecs[i])); // lambda_i = (v_i^dag M v_i) / ||v_i||
    Complex Cm1(-1.0, 0.0);
    blas::caxpby(lambda[i], *eVecs[i], Cm1, *w); // w = lambda_i*v_i - A*v_i
    r[i] = sqrt(blas::norm2(*w)); // r = ||w||
  }

  if(eigParams->diracType == MUGIQ_EIG_OPERATOR_MdagM || eigParams->diracType == MUGIQ_EIG_OPERATOR_MMdag){
    std::vector<double> &sigma = *eVals_sigma;    
    for(int i=0; i<eigParams->nEv; i++) sigma[i] = sqrt(lambda[i].real());
  }
  
  delete w;
}

void Eigsolve_Mugiq::printEvals(){

  printfQuda("\nEigsolve_Mugiq - Eigenvalues:\n");
  
  std::vector<Complex> &evals_quda = *eVals_quda;
  std::vector<Complex> &evals = *eVals;
  std::vector<double> &res = *evals_res;
  for(int i=0;i<eigParams->nEv;i++)
    printfQuda("Mugiq-Quda: Eval[%04d] = %+.16e %+.16e , %+.16e %+.16e , Residual = %+.16e\n", i,
               evals[i].real(), evals[i].imag(), evals_quda[i].real(), evals_quda[i].imag(), res[i]);

  if(eigParams->diracType == MUGIQ_EIG_OPERATOR_MdagM || eigParams->diracType == MUGIQ_EIG_OPERATOR_MMdag){
    std::vector<double> &sigma = *eVals_sigma;    
    printfQuda("\n");
    for(int i=0;i<eigParams->nEv;i++)
      printfQuda("Mugiq-Quda: Sigma[%04d] = %+.16e\n", i, sigma[i]);
  }
  
}
