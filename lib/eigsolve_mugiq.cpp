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
  eVals_quda(nullptr),
  eVals(nullptr),
  evals_res(nullptr),
  computeCoarse(computeCoarse_)
{
  if(!mg_env->mgInit) errorQuda("%s: Multigrid environment must be initialized before Eigensolver.\n", __func__);
  
  cudaGaugeField *gauge = checkGauge(invParams);
  const int *X = gauge->X(); //- The Lattice Size

  if(computeCoarse){
    int nInterLevels = mg_env->nInterLevels;
    
    // Create the Coarse Dirac operator
    dirac = mg_env->diracCoarse; // This is diracCoarseResidual of the QUDA MG class
    if(typeid(*dirac) != typeid(DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");

    ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
    QudaPrecision coarsePrec = invParams->cuda_prec;
         
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
  else{
    //-The fine Dirac operator
    // This is diracResidual of the QUDA MG class.
    // Equivalently: dirac = mg_env->mg_solver->mgParam->matResidual.Expose() = mg_env->mg_solver->d
    dirac = mg_env->mg_solver->d; 

    //- Allocate fine eigenvectors
    ColorSpinorParam cpuParam(NULL, *invParams, X, invParams->solution_type, invParams->input_location);
    ColorSpinorParam cudaParam(cpuParam);
    cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaParam.setPrecision(eigParams->QudaEigParams->cuda_prec_ritz, eigParams->QudaEigParams->cuda_prec_ritz, true);

    for(int i=0;i<eigParams->nEv;i++)
      eVecs.push_back(ColorSpinorField::Create(cudaParam));
  }

  
  //- Determine and create the Dirac matrix whose eigenvalues will be computed
  eigParams->diracType = determineEigOperator(eigParams->QudaEigParams);

  if      (eigParams->diracType == MUGIQ_EIG_OPERATOR_M)      mat = new DiracM(*dirac);	  
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_Mdag)   mat = new DiracMdag(*dirac); 
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_MdagM)  mat = new DiracMdagM(*dirac);
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_MMdag)  mat = new DiracMMdag(*dirac);
  else errorQuda("%s: Unsupported Dirac operator type\n", __func__);

  
  //- Allocate the eigenvalues
  eVals_quda = new std::vector<Complex>(eigParams->nEv, 0.0); // These come from the QUDA eigensolver
  eVals      = new std::vector<Complex>(eigParams->nEv, 0.0); // These are computed from the Eigsolve_Mugiq class
  evals_res  = new std::vector<double>(eigParams->nEv, 0.0);  // The eigenvalues residual
  
  makeChecks();

  eigInit = MUGIQ_BOOL_TRUE;
}


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
  eVals_quda(nullptr),
  eVals(nullptr),
  evals_res(nullptr),
  computeCoarse(MUGIQ_BOOL_INVALID)
{

  //- Whether we are running for the "full" or even-odd preconditioned Operator
  bool pc_solve = (invParams->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (invParams->solve_type == QUDA_NORMOP_PC_SOLVE) ||
    (invParams->solve_type == QUDA_NORMERR_PC_SOLVE);
  
  //- Create the Dirac operator
  DiracParam diracParam;
  setDiracParam(diracParam, invParams, pc_solve);
  dirac = Dirac::create(diracParam);

  //- Determine and create the Dirac matrix whose eigenvalues will be computed
  eigParams->diracType = determineEigOperator(eigParams->QudaEigParams);

  if      (eigParams->diracType == MUGIQ_EIG_OPERATOR_M)      mat = new DiracM(*dirac);	  
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_Mdag)   mat = new DiracMdag(*dirac); 
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_MdagM)  mat = new DiracMdagM(*dirac);
  else if (eigParams->diracType == MUGIQ_EIG_OPERATOR_MMdag)  mat = new DiracMMdag(*dirac);
  else errorQuda("%s: Unsupported Dirac operator type\n", __func__);

  
  cudaGaugeField *gauge = checkGauge(invParams);
  const int *X = gauge->X(); //- The Lattice Size

  ColorSpinorParam cpuParam(NULL, *invParams, X, invParams->solution_type, invParams->input_location);
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(eigParams->QudaEigParams->cuda_prec_ritz, eigParams->QudaEigParams->cuda_prec_ritz, true);

  //- Allocate the eigenvectors
  for(int i=0;i<eigParams->nEv;i++)
    eVecs.push_back(ColorSpinorField::Create(cudaParam));

  //- Allocate the eigenvalues
  eVals_quda = new std::vector<Complex>(eigParams->nEv, 0.0); // These come from the QUDA eigensolver
  eVals      = new std::vector<Complex>(eigParams->nEv, 0.0); // These are computed from the Eigsolve_Mugiq class
  evals_res  = new std::vector<double>(eigParams->nEv, 0.0);  // The eigenvalues residual

  makeChecks();

  eigInit = MUGIQ_BOOL_TRUE;
}


Eigsolve_Mugiq::~Eigsolve_Mugiq(){  
  for(int i=0;i<eigParams->nEv;i++) delete eVecs[i];
  delete eVals_quda;
  delete eVals;
  delete evals_res;
  
  if(mat) delete mat;
  mat = nullptr;
  
  if(useMGenv){
    //- (dirac deletion is taken care by mg_solver destructor in this case)
    int nTmp = static_cast<int>(tmpCSF.size());
    for(int i=0;i<nTmp;i++) if(tmpCSF[i]) delete tmpCSF[i];    
  }
  else{
    if(dirac) delete dirac;
    dirac = nullptr;
  } 
  
  eigInit = MUGIQ_BOOL_FALSE;
}



MuGiqEigOperator Eigsolve_Mugiq::determineEigOperator(QudaEigParam *eigParams_){

  MuGiqEigOperator dType = MUGIQ_EIG_OPERATOR_INVALID;
  
  if (!eigParams_->use_norm_op && !eigParams_->use_dagger)     dType = MUGIQ_EIG_OPERATOR_M;
  else if (!eigParams_->use_norm_op && eigParams_->use_dagger) dType = MUGIQ_EIG_OPERATOR_Mdag;
  else if (eigParams_->use_norm_op && !eigParams_->use_dagger) dType = MUGIQ_EIG_OPERATOR_MdagM;
  else if (eigParams_->use_norm_op && eigParams_->use_dagger)  dType = MUGIQ_EIG_OPERATOR_MMdag;
  else errorQuda("%s: Cannot determine Dirac Operator type from eigParams\n", __func__);

  return dType;
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
}
