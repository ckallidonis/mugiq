#ifndef _EISGOLVE_MUGIQ_H
#define _EIGSOLVE_MUGIQ_H

#include <quda.h>
#include <mg_mugiq.h>

class Eigsolve_Mugiq {

private:

  QudaEigParam *eigParams;     // Eigensolver parameter

  QudaInvertParam *invParams;  // Inverter parameters

  MG_Mugiq *mg;                // Multigrid object

  quda::TimeProfile &eig_profile; // Used for profiling
  
public:
  Eigsolve_Mugiq(MG_Mugiq *mg_, QudaEigParam *eigParams_, quda::TimeProfile &profile_);
  ~Eigsolve_Mugiq();

  /** @brief Perform basic checks based on parameter structure input values
   */
  void makeChecks();

}; // class Eigsolve_Mugiq 



#endif // _EIGSOLVE_MUGIQ_H
