#ifndef _MG_MUGIQ_H
#define _MG_MUGIQ_H

#include <multigrid.h> //- The QUDA MG header file

using namespace quda;

struct MG_Mugiq {

  multigrid_solver *mg_solver; // The multigrid structure

  QudaMultigridParam *mgParams; // Multigrid parameters

  bool mgInit; // Initialization switch

  Dirac *diracCoarse; // The Coarse Dirac operator

  TimeProfile &profile; // Profiling
  
  MG_Mugiq(QudaMultigridParam *mgParams_, TimeProfile &profile_); // Constructor
  
  virtual ~MG_Mugiq(){
    if (mg_solver) delete mg_solver;
    mg_solver = nullptr;

    diracCoarse = nullptr;

    mgInit = false;
  }
  

}; //- Struct MG_Mugiq


#endif // _MG_MUGIQ_H
