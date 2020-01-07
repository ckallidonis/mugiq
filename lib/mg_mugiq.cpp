#include <mg_mugiq.h>


MG_Mugiq::MG_Mugiq(QudaMultigridParam *mgParams_, QudaEigParam *eigParams_, TimeProfile &profile_) :
  mgParams(mgParams_),
  eigParams(eigParams_),
  mgInit(false),
  diracCoarse(nullptr),
  matCoarse(nullptr),
  mg_profile(profile_)
{
  mg_solver = new multigrid_solver(*mgParams, mg_profile);

  diracCoarse = mg_solver->mg->getDiracCoarse();
  if(typeid(*diracCoarse) != typeid(DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");

  // Create the Coarse Dirac operator
  // That's the operator whose eigenvalues will be computed
  if (!eigParams->use_norm_op && !eigParams->use_dagger)     matCoarse = new DiracM(*diracCoarse);
  else if (!eigParams->use_norm_op && eigParams->use_dagger) matCoarse = new DiracMdag(*diracCoarse);
  else if (eigParams->use_norm_op && !eigParams->use_dagger) matCoarse = new DiracMdagM(*diracCoarse);
  else if (eigParams->use_norm_op && eigParams->use_dagger)  matCoarse = new DiracMMdag(*diracCoarse);
  
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
