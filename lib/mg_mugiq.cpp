#include <mg_mugiq.h>


MG_Mugiq::MG_Mugiq(QudaMultigridParam *mgParams_, TimeProfile &profile_) :
  mgParams(mgParams_),
  mgInit(false),
  diracCoarse(nullptr),
  profile(profile_)
{
  mg_solver = new multigrid_solver(*mgParams, profile);

  diracCoarse = mg_solver->mg->getDiracCoarse();
  if(typeid(*diracCoarse) != typeid(DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");
  
  mgInit = true;
}
