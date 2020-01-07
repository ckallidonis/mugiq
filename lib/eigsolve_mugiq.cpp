#include <eigsolve_mugiq.h>
#include <util_quda.h>

Eigsolve_Mugiq::Eigsolve_Mugiq(MG_Mugiq *mg_, QudaEigParam *eigParams_, quda::TimeProfile &profile_) :
  eigParams(eigParams_),
  invParams(eigParams_->invert_param),
  mg(mg_),
  eig_profile(profile_)
{
  if(!mg->mgInit) errorQuda("%s: MG_Mugiq must be initialized before proceeding with eigensolver.\n", __func__);
  makeChecks();
}

Eigsolve_Mugiq::~Eigsolve_Mugiq(){
}

void Eigsolve_Mugiq::makeChecks(){

  if (invParams->dslash_type != QUDA_WILSON_DSLASH && invParams->dslash_type != QUDA_CLOVER_WILSON_DSLASH)
    errorQuda("%s: Supports only Wilson and Wilson-Clover operators for now!\n", __func__);

  // No polynomial acceleration on non-symmetric matrices
  if (eigParams->use_poly_acc && !eigParams->use_norm_op && !(invParams->dslash_type == QUDA_LAPLACE_DSLASH))
    errorQuda("%s: Polynomial acceleration with non-symmetric matrices not supported", __func__);
}
