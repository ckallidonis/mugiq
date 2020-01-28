#include <eigsolve_mugiq.h>
#include <util_quda.h>
#include <mugiq_internal.h>
#include <eigensolve_quda.h>

Eigsolve_Mugiq::Eigsolve_Mugiq(MG_Mugiq *mg_env_,
			       QudaEigParam *eigParams_,
			       TimeProfile *eigProfile_,
			       bool computeCoarse_) :
  mg_env(mg_env_),
  eigInit(false),
  mgEigsolve(false),
  computeCoarse(computeCoarse_),
  mgParams(mg_env->mgParams),
  eigParams(eigParams_),
  invParams(eigParams->invert_param),
  eigProfile(eigProfile_),
  dirac(nullptr),
  mat(nullptr),
  pc_solve(false),
  nConv(eigParams_->nConv)
{
  if(!mg_env->mgInit) errorQuda("%s: Multigrid environment must be initialized before Eigensolver.\n", __func__);
  
  cudaGaugeField *gauge = checkGauge(invParams);
  const int *X = gauge->X(); //- The Lattice Size

  if(computeCoarse){
    int nCoarseLevels = mgParams->n_level-1;
    int nextCoarse = nCoarseLevels - 1;
    
    // Create the Coarse Dirac operator
    dirac = mg_env->diracCoarse; // This is diracCoarseResidual of the QUDA MG class
    if(typeid(*dirac) != typeid(DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");

    ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
    QudaPrecision coarsePrec = invParams->cuda_prec;
         
    //-Create coarse fields and allocate coarse eigenvectors recursively
    tmpCSF.push_back(ColorSpinorField::Create(csParam)); //- tmpCSF[0] is a fine field
    for(int lev=0;lev<nextCoarse;lev++){
      tmpCSF.push_back(tmpCSF[lev]->CreateCoarse(mgParams->geo_block_size[lev],
						 mgParams->spin_block_size[lev],
						 mgParams->n_vec[lev],
						 coarsePrec,
						 mgParams->setup_location[lev+1]));
    }//-lev

    for(int i=0;i<nConv;i++){
      eVecs.push_back(tmpCSF[nextCoarse]->CreateCoarse(mgParams->geo_block_size[nextCoarse],
						       mgParams->spin_block_size[nextCoarse],
						       mgParams->n_vec[nextCoarse],
						       coarsePrec,
						       mgParams->setup_location[nextCoarse+1]));
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
    cudaParam.setPrecision(eigParams->cuda_prec_ritz, eigParams->cuda_prec_ritz, true);

    for(int i=0;i<nConv;i++)
      eVecs.push_back(ColorSpinorField::Create(cudaParam));
  }

  
  //- That's the Dirac matrix whose eigenvalues will be computed
  if (!eigParams->use_norm_op && !eigParams->use_dagger)     mat = new DiracM(*dirac);
  else if (!eigParams->use_norm_op && eigParams->use_dagger) mat = new DiracMdag(*dirac);
  else if (eigParams->use_norm_op && !eigParams->use_dagger) mat = new DiracMdagM(*dirac);
  else if (eigParams->use_norm_op && eigParams->use_dagger)  mat = new DiracMMdag(*dirac);

  
  //- Allocate the eigenvalues
  eVals_quda = new std::vector<Complex>(nConv, 0.0); // These come from the QUDA eigensolver
  eVals      = new std::vector<Complex>(nConv, 0.0); // These are computed from the Eigsolve_Mugiq class
  evals_res  = new std::vector<double>(nConv, 0.0);  // The eigenvalues residual
  
  makeChecks();

  eigInit = true;
  mgEigsolve = true;
}


Eigsolve_Mugiq::Eigsolve_Mugiq(QudaEigParam *eigParams_, TimeProfile *profile_) :
  eigInit(false),
  mgEigsolve(false),
  computeCoarse(false),
  eigParams(eigParams_),
  invParams(eigParams_->invert_param),
  eigProfile(profile_),
  dirac(nullptr),
  mat(nullptr),
  pc_solve(false),
  nConv(eigParams_->nConv)
{

  //- Whether we are running for the "full" or even-odd preconditioned Operator
  pc_solve = (invParams->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (invParams->solve_type == QUDA_NORMOP_PC_SOLVE) ||
    (invParams->solve_type == QUDA_NORMERR_PC_SOLVE);
  
  //- Create the Dirac operator
  DiracParam diracParam;
  setDiracParam(diracParam, invParams, pc_solve);
  dirac = Dirac::create(diracParam);

  if (!eigParams->use_norm_op && !eigParams->use_dagger)     mat = new DiracM(*dirac);
  else if (!eigParams->use_norm_op && eigParams->use_dagger) mat = new DiracMdag(*dirac);
  else if (eigParams->use_norm_op && !eigParams->use_dagger) mat = new DiracMdagM(*dirac);
  else if (eigParams->use_norm_op && eigParams->use_dagger)  mat = new DiracMMdag(*dirac);
  
  cudaGaugeField *gauge = checkGauge(invParams);
  const int *X = gauge->X(); //- The Lattice Size

  ColorSpinorParam cpuParam(NULL, *invParams, X, invParams->solution_type, invParams->input_location);
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(eigParams->cuda_prec_ritz, eigParams->cuda_prec_ritz, true);

  //- Allocate the eigenvectors
  for(int i=0;i<nConv;i++)
    eVecs.push_back(ColorSpinorField::Create(cudaParam));

  //- Allocate the eigenvalues
  eVals_quda = new std::vector<Complex>(nConv, 0.0); // These come from the QUDA eigensolver
  eVals      = new std::vector<Complex>(nConv, 0.0); // These are computed from the Eigsolve_Mugiq class
  evals_res  = new std::vector<double>(nConv, 0.0);  // The eigenvalues residual

  makeChecks();

  eigInit = true;
  mgEigsolve = false;
}


Eigsolve_Mugiq::~Eigsolve_Mugiq(){  
  for(int i=0;i<nConv;i++) delete eVecs[i];
  delete eVals_quda;
  delete eVals;
  delete evals_res;
  
  if(mat) delete mat;
  mat = nullptr;
  
  if(mgEigsolve){
    //- (dirac deletion is taken care by mg_solver destructor in this case)
    int nTmp = static_cast<int>(tmpCSF.size());
    for(int i=0;i<nTmp;i++) if(tmpCSF[i]) delete tmpCSF[i];    
  }
  else{
    if(dirac) delete dirac;
    dirac = nullptr;
  } 
  
  eigInit = false;
  mgEigsolve = false;
}

void Eigsolve_Mugiq::makeChecks(){

  if(!mat) errorQuda("%s: Dirac operator is not defined.\n", __func__);
  
  if (invParams->dslash_type != QUDA_WILSON_DSLASH && invParams->dslash_type != QUDA_CLOVER_WILSON_DSLASH)
    errorQuda("%s: Supports only Wilson and Wilson-Clover operators for now!\n", __func__);

  // No polynomial acceleration on non-symmetric matrices
  if (eigParams->use_poly_acc && !eigParams->use_norm_op && !(invParams->dslash_type == QUDA_LAPLACE_DSLASH))
    errorQuda("%s: Polynomial acceleration with non-symmetric matrices not supported", __func__);
}


void Eigsolve_Mugiq::computeEvecs(){

  if(!eigInit) errorQuda("%s: Eigsolve_Mugiq must be initialized first.\n", __func__);

  //- Perform eigensolve
  EigenSolver *eigSolve = EigenSolver::create(eigParams, *mat, *eigProfile);
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
  
  for(int i=0; i<nConv; i++){
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
  for(int i=0;i<nConv;i++)
    printfQuda("Mugiq-Quda: Eval[%04d] = %+.16e %+.16e , %+.16e %+.16e , Residual = %+.16e\n", i,
               evals[i].real(), evals[i].imag(), evals_quda[i].real(), evals_quda[i].imag(), res[i]);  
}
