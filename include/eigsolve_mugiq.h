#ifndef _EISGOLVE_MUGIQ_H
#define _EIGSOLVE_MUGIQ_H

#include <quda.h>
#include <mg_mugiq.h>
#include <color_spinor_field.h>

using namespace quda;

class Eigsolve_Mugiq {

private:

  bool eigInit;                   // Initialization switch
  bool mgEigsolve;                // MG eigsolve switch
  QudaEigParam *eigParams;        // Eigensolver parameter
  QudaInvertParam *invParams;     // Inverter parameters
  TimeProfile &eig_profile; // Used for profiling

  MG_Mugiq *mg;   // Multigrid object

  const Dirac *dirac;
  DiracMatrix *mat; // The Dirac operator whose eigenpairs we are computing

  bool pc_solve; // Whether we are running for the "full" or even-odd preconditioned Operator
  
  std::vector<ColorSpinorField *> eVecs; // Eigenvectors
  std::vector<Complex> *eVals; // Eigenvalues
  std::vector<Complex> *eVals_loc; // Local Eigenvalues, computed within the class

  std::vector<double> *evals_res;
  
  int nConv; // Number of eigenvectors we want
  
public:
  Eigsolve_Mugiq(MG_Mugiq *mg_, QudaEigParam *eigParams_, TimeProfile &profile_);
  Eigsolve_Mugiq(QudaEigParam *eigParams_, TimeProfile &profile_);
  ~Eigsolve_Mugiq();

  /** @brief Perform basic checks based on parameter structure input values
   */
  void makeChecks();
  
  /** @brief Compute eigenvectors of Dirac operator
   */
  void computeEvecs();

  /** @brief Compute eigenvalues
   */
  void computeEvals();

  /** @brief Compute eigenvalues
   */
  void printEvals();

  /** @brief Accessor to get the eigenvalues outside of the class
   */
  std::vector<Complex>* getEvals(){ return eVals;}

  /** @brief Accessor to get the local eigenvalues outside of the class
   */
  std::vector<Complex>* getEvals_loc(){ return eVals_loc;}
  
  /** @brief Accessor to get the residual of the computed eigenvalues
   */
  std::vector<double>* getEvalsRes(){ return evals_res;}


  
  
}; // class Eigsolve_Mugiq 



#endif // _EIGSOLVE_MUGIQ_H
