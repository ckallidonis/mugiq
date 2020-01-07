#ifndef _EISGOLVE_MUGIQ_H
#define _EIGSOLVE_MUGIQ_H

#include <quda.h>

class Eigsolve_Mugiq {

private:
  QudaEigParam *eigParams;
  QudaInvertParam *invParams;
  
public:
  Eigsolve_Mugiq(QudaEigParam *eigParams_);
  ~Eigsolve_Mugiq();

  /** @brief Perform basic checks based on parameter structure input values
   */
  void makeChecks();

}; // class Eigsolve_Mugiq 



#endif // _EIGSOLVE_MUGIQ_H
