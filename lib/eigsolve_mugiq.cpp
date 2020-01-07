#include <eigsolve_mugiq.h>

Eigsolve_Mugiq::Eigsolve_Mugiq(QudaEigParam *eigParams_) :
  eigParams(eigParams_),
  invParams(eigParams_->invert_param)
{
}

Eigsolve_Mugiq::~Eigsolve_Mugiq(){
}

void makeChecks(){
}
