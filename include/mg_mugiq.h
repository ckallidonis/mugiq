#ifndef _MG_MUGIQ_H
#define _MG_MUGIQ_H

#include <multigrid.h> //- The QUDA MG header file

using namespace quda;

struct MG_Mugiq {

  multigrid_solver *mg_solver; // The multigrid structure

  QudaMultigridParam *mgParams; // Multigrid parameters

  bool mgInit; // Initialization switch

  Dirac *diracCoarse; // The Coarse Dirac operator

  Transfer *transfer;       // Transfer Operator at coarsest Level
  Transfer *transferCoarse; // Transfer Operator
  
  TimeProfile &profile; // Profiling
  
  MG_Mugiq(QudaMultigridParam *mgParams_, TimeProfile &profile_) :
    mgParams(mgParams_),
    mgInit(false),
    diracCoarse(nullptr),
    transfer(nullptr),
    //    transferLevel(nullptr),
    profile(profile_)
  {
    mg_solver = new multigrid_solver(*mgParams, profile);
    
    diracCoarse = mg_solver->mg->getDiracCoarse();
    if(typeid(*diracCoarse) != typeid(DiracCoarse)) errorQuda("The Coarse Dirac operator must not be preconditioned!\n");

    if(mgParams->n_level > 3) errorQuda("Unsupported MG levels %d\n", mgParams->n_level);

    if(mgParams->n_level == 2)
      transferCoarse = mg_solver->mg->getTransfer();

    if(mgParams->n_level > 2){
      transfer = mg_solver->mg->getTransfer();
      transferCoarse = mg_solver->mg->getTransferCoarse();
    }
        
    mgInit = true;
  }
  
  virtual ~MG_Mugiq(){
    if (mg_solver) delete mg_solver;
    mg_solver = nullptr;

    diracCoarse = nullptr;

    mgInit = false;
  }
  

}; //- Struct MG_Mugiq


#endif // _MG_MUGIQ_H
