#ifndef _MG_MUGIQ_H
#define _MG_MUGIQ_H

#include <multigrid.h> //- The QUDA MG header file

using namespace quda;

struct MG_Mugiq {

  multigrid_solver *mg_solver; // The multigrid structure

  QudaMultigridParam *mgParams; // Multigrid parameters

  bool mgInit; // Initialization switch

  Dirac *diracCoarse; // The Coarse Dirac operator

  Transfer *transfer[QUDA_MAX_MG_LEVEL-1];       // Transfer Operator at coarsest Level

  int nCoarseLevels;   // Number of all Coarse Levels
  int nInterLevels;    // Number of intermediate levels between finest and coarsest
  int maxCoarseLevels; // Maximum number of coarse levels
  
  TimeProfile &profile; // Profiling
  
  MG_Mugiq(QudaMultigridParam *mgParams_, TimeProfile &profile_) :
    mgParams(mgParams_),
    mgInit(false),
    diracCoarse(nullptr),
    nCoarseLevels(mgParams->n_level-1),
    nInterLevels(mgParams->n_level-2),
    maxCoarseLevels(QUDA_MAX_MG_LEVEL-1),
    profile(profile_)
  {
    mg_solver = new multigrid_solver(*mgParams, profile);
    
    diracCoarse = mg_solver->mg->getDiracCoarse();
    if(typeid(*diracCoarse) != typeid(DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");

    if(mgParams->n_level == 2){
      transfer[0] = mg_solver->mg->getTransferFinest();
      for(int i=1;i<maxCoarseLevels;i++) transfer[i] = nullptr;
    }
    else if(mgParams->n_level == 3){
      transfer[0] = mg_solver->mg->getTransferFinest();
      transfer[1] = mg_solver->mg->getTransferCoarsest();
      transfer[2] = nullptr;
    }
    else if(mgParams->n_level == 4){
      transfer[0] = mg_solver->mg->getTransferFinest();
      transfer[1] = mg_solver->mg->getTransferCoarse();
      transfer[2] = mg_solver->mg->getTransferCoarsest();
    }

    mgInit = true;
  }

  
  virtual ~MG_Mugiq(){
    if (mg_solver) delete mg_solver;
    mg_solver = nullptr;
    for(int i=0;i<maxCoarseLevels;i++) transfer[i] = nullptr;    
    diracCoarse = nullptr;
    mgInit = false;
  }
  
}; //- Struct MG_Mugiq


#endif // _MG_MUGIQ_H
