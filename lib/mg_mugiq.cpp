#include <mg_mugiq.h>


MG_Mugiq::MG_Mugiq(QudaMultigridParam *mgParams_, QudaEigParam *eigParams_, quda::TimeProfile &profile_) :
  mgParams(mgParams_),
  eigParams(eigParams_),
  mgInit(false),
  diracCoarse(nullptr),
  matCoarse(nullptr)
{
  mg_solver = new quda::multigrid_solver(*mgParams, profile_);

  diracCoarse = mg_solver->mg->getDiracCoarse();
  if(typeid(*diracCoarse) != typeid(quda::DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");

  // Create the Coarse Dirac operator
  // That's the operator whose eigenvalues will be computed
  if (!eigParams->use_norm_op && !eigParams->use_dagger)     matCoarse = new quda::DiracM(*diracCoarse);
  else if (!eigParams->use_norm_op && eigParams->use_dagger) matCoarse = new quda::DiracMdag(*diracCoarse);
  else if (eigParams->use_norm_op && !eigParams->use_dagger) matCoarse = new quda::DiracMdagM(*diracCoarse);
  else if (eigParams->use_norm_op && eigParams->use_dagger)  matCoarse = new quda::DiracMMdag(*diracCoarse);
  
  mgInit = true;
}

MG_Mugiq::~MG_Mugiq(){

  if(matCoarse) delete matCoarse;
  matCoarse = nullptr;
  diracCoarse = nullptr;

  if (mg_solver) delete mg_solver;
  mg_solver = nullptr;
  
  mgInit = false;
}
