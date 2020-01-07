#ifndef _MG_MUGIQ_H
#define _MG_MUGIQ_H

#include <multigrid.h> //- The QUDA MG header file
#include <eigsolve_mugiq.h>

class MG_Mugiq {

  friend class Eigsolve_Mugiq;
  
private:
  
  quda::multigrid_solver *mg_solver; // The multigrid structure

  QudaMultigridParam *mgParams; // Multigrid parameters

  QudaEigParam *eigParams; // Eigsolve parameters

  bool mgInit; // Initialization switch

  quda::Dirac *diracCoarse; // The Coarse Dirac operator

  quda::DiracMatrix *matCoarse; // Wrapper for the Coarse operator
  
  
public:
  
  MG_Mugiq(QudaMultigridParam *mgParams_, QudaEigParam *eigParams_, quda::TimeProfile &profile_); // Constructor
  
  virtual ~MG_Mugiq(); // Destructor
  

}; //- Class MG_Mugiq


#endif // _MG_MUGIQ_H
