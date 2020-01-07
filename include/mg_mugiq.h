#ifndef _MG_MUGIQ_H
#define _MG_MUGIQ_H

#include <multigrid.h> //- The QUDA MG header file

using namespace quda;

class MG_Mugiq {

  friend class Eigsolve_Mugiq;
  
private:
  
  multigrid_solver *mg_solver; // The multigrid structure

  QudaMultigridParam *mgParams; // Multigrid parameters

  QudaEigParam *eigParams; // Eigsolve parameters

  bool mgInit; // Initialization switch

  Dirac *diracCoarse; // The Coarse Dirac operator

  DiracMatrix *matCoarse; // Wrapper for the Coarse operator

  TimeProfile &mg_profile; // Profiling
  
public:
  
  MG_Mugiq(QudaMultigridParam *mgParams_, QudaEigParam *eigParams_, TimeProfile &profile_); // Constructor
  
  virtual ~MG_Mugiq(); // Destructor
  

}; //- Class MG_Mugiq


#endif // _MG_MUGIQ_H
