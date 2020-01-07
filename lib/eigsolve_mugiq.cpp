#include <mg_mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_quda.h>
#include <mugiq_internal.h>
#include <eigensolve_quda.h>

Eigsolve_Mugiq::Eigsolve_Mugiq(MG_Mugiq *mg_, QudaEigParam *eigParams_, TimeProfile &profile_) :
  eigInit(false),
  eigParams(eigParams_),
  invParams(eigParams_->invert_param),
  eig_profile(profile_),
  mg(mg_),
  mat(nullptr),
  nConv(eigParams_->nConv)
{
  
  if(!mg->mgInit) errorQuda("%s: MG_Mugiq must be initialized before proceeding with eigensolver.\n", __func__);

  //- The Dirac operator
  mat = mg->matCoarse;
  
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

  //- Allocate the eigenvalues These are the eigenvalues coming from the Quda eigensolver
  eVals = new std::vector<Complex>(nConv, 0.0);

  makeChecks();

  eigInit = true;
}

Eigsolve_Mugiq::~Eigsolve_Mugiq(){
  for(int i=0;i<nConv;i++) delete eVecs[i];
  delete eVals;
  eigInit = false;
}

void Eigsolve_Mugiq::makeChecks(){

  if(!mat) errorQuda("%s: Dirac operator is not defined.\n", __func__);
  
  if (invParams->dslash_type != QUDA_WILSON_DSLASH && invParams->dslash_type != QUDA_CLOVER_WILSON_DSLASH)
    errorQuda("%s: Supports only Wilson and Wilson-Clover operators for now!\n", __func__);

  // No polynomial acceleration on non-symmetric matrices
  if (eigParams->use_poly_acc && !eigParams->use_norm_op && !(invParams->dslash_type == QUDA_LAPLACE_DSLASH))
    errorQuda("%s: Polynomial acceleration with non-symmetric matrices not supported", __func__);
}


void Eigsolve_Mugiq::computeCoarseEvecs(){

  if(!eigInit) errorQuda("%s: Eigsolve_Mugiq must be initialized first.\n", __func__);

  //- Perform eigensolve
  EigenSolver *eigSolve = EigenSolver::create(eigParams, *mat, eig_profile);
  (*eigSolve)(eVecs, *eVals);

  delete eigSolve;
}
