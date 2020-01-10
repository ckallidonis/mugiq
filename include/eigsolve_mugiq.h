#ifndef _EISGOLVE_MUGIQ_H
#define _EIGSOLVE_MUGIQ_H

#include <quda.h>
#include <multigrid.h>           //- The QUDA MG header file
#include <color_spinor_field.h>  //- From QUDA
#include <mg_mugiq.h>

using namespace quda;

class Eigsolve_Mugiq {

private:

  bool eigInit;                   // Initialization switch
  bool mgEigsolve;                // MG eigsolve switch

  // Whether we are computing eigenpairs of the coarse Dirac operator
  // (significant only when using Multigrid, default true) 
  bool computeCoarse; 

  QudaMultigridParam *mgParams;   // Multigrid parameter structure
  QudaEigParam *eigParams;        // Eigensolver parameter structure
  QudaInvertParam *invParams;     // Inverter parameter structure

  TimeProfile *mg_profile; //  Used for profiling
  TimeProfile *eig_profile; // Used for profiling

  multigrid_solver *mg_solver; // The multigrid structure

  const Dirac *dirac;
  DiracMatrix *mat; // The Dirac operator whose eigenpairs we are computing

  bool pc_solve; // Whether we are running for the "full" or even-odd preconditioned Operator

  
  std::vector<ColorSpinorField *> eVecs; // Eigenvectors
  std::vector<Complex> *eVals; // Eigenvalues
  std::vector<Complex> *eVals_loc; // Local Eigenvalues, computed within the class

  std::vector<ColorSpinorField *> tmpCSF; // Temporary field(s)
  
  std::vector<double> *evals_res;
  
  int nConv; // Number of eigenvectors we want
  
public:
  Eigsolve_Mugiq(QudaMultigridParam *mgParams_, TimeProfile *mg_profile_,
		 QudaEigParam *eigParams_,      TimeProfile *eig_profile_,
		 bool computeCoarse_=true);

  Eigsolve_Mugiq(QudaEigParam *eigParams_, TimeProfile *profile_);
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
