#ifndef _EISGOLVE_MUGIQ_H
#define _EIGSOLVE_MUGIQ_H

#include <quda.h>
#include <mg_mugiq.h>
#include <color_spinor_field.h>

using namespace quda;

class Eigsolve_Mugiq {

private:

  bool eigInit;                   // Initialization switch
  QudaEigParam *eigParams;        // Eigensolver parameter
  QudaInvertParam *invParams;     // Inverter parameters
  TimeProfile &eig_profile; // Used for profiling

  MG_Mugiq *mg;   // Multigrid object

  std::vector<ColorSpinorField *> coarseEvecs; // Eigenvectors of the coarse Dirac operator
  std::vector<Complex> *coarseEvals; // Eigenvalues of the coarse Dirac operator
  
  int nConv; // Number of eigenvectors we want
  
public:
  Eigsolve_Mugiq(MG_Mugiq *mg_, QudaEigParam *eigParams_, TimeProfile &profile_);
  ~Eigsolve_Mugiq();

  /** @brief Perform basic checks based on parameter structure input values
   */
  void makeChecks();

  
}; // class Eigsolve_Mugiq 



#endif // _EIGSOLVE_MUGIQ_H
