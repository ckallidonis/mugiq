#include <eigsolve_mugiq.h>
#include <util_quda.h>

Eigsolve_Mugiq::Eigsolve_Mugiq(QudaEigParam *eigParams_) :
  eigParams(eigParams_),
  invParams(eigParams_->invert_param)
{
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
