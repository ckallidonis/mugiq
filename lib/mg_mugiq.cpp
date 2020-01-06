#include <mg_mugiq.h>


MG_Mugiq::MG_Mugiq(QudaMultigridParam *param_, quda::TimeProfile &profile_) :
  param(param_)
{
  mg_solver = new quda::multigrid_solver(*param, profile_);
}

MG_Mugiq::~MG_Mugiq(){
  if (mg_solver) delete mg_solver;
}
