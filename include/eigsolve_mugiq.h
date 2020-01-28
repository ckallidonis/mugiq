#ifndef _EISGOLVE_MUGIQ_H
#define _EIGSOLVE_MUGIQ_H

#include <quda.h>
#include <multigrid.h>           //- The QUDA MG header file
#include <color_spinor_field.h>  //- From QUDA
#include <mg_mugiq.h>

using namespace quda;

class Eigsolve_Mugiq {

private:

  MG_Mugiq *mg_env; // The Multigrid environment

  bool eigInit;                   // Initialization switch
  bool mgEigsolve;                // MG eigsolve switch

  // Whether we are computing eigenpairs of the coarse Dirac operator
  // (significant only when using Multigrid, default true) 
  bool computeCoarse; 

  QudaMultigridParam *mgParams;   // Multigrid parameter structure
  QudaEigParam *eigParams;        // Eigensolver parameter structure
  QudaInvertParam *invParams;     // Inverter parameter structure

  TimeProfile *eigProfile; // Used for profiling
  
  const Dirac *dirac;
  DiracMatrix *mat; // The Dirac operator whose eigenpairs we are computing

  bool pc_solve; // Whether we are running for the "full" or even-odd preconditioned Operator

  
  std::vector<ColorSpinorField *> eVecs; // Eigenvectors
  std::vector<Complex> *eVals_quda; // Eigenvalues from the Quda eigensolver
  std::vector<Complex> *eVals; // Eigenvalues computed within the Eigsolve_Mugiq class

  std::vector<ColorSpinorField *> tmpCSF; // Temporary field(s)
  
  std::vector<double> *evals_res;
  
  int nConv; // Number of eigenvectors we want
  
public:
  Eigsolve_Mugiq(MG_Mugiq *mg_env_,
		 QudaEigParam *eigParams_, TimeProfile *eigProfile_,
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

  /** @brief Accessor to get the eigenvectors outside of the class
   */
  std::vector<ColorSpinorField *> &getEvecs(){ return eVecs;}
  
  /** @brief Accessor to get the Quda eigenvalues outside of the class
   */
  std::vector<Complex>* getEvalsQuda(){ return eVals_quda;}
  
  /** @brief Accessor to get the Eigsolve_Mugiq eigenvalues outside of the class
   */
  std::vector<Complex> *getEvals(){ return eVals;}
  
  /** @brief Accessor to get the residual of the computed eigenvalues
   */
  std::vector<double>* getEvalsRes(){ return evals_res;}

  /** @brief Accessor to get the Multigrid environment structure
   */
  MG_Mugiq* getMGEnv(){ return mg_env;}
  
  /** @brief Accessor to get the eigsolve parameter structure
   */
  QudaEigParam* getEigParams(){ return eigParams;}
  
  /** @brief Accessor to get the invert parameter structure
   */
  QudaInvertParam* getInvParams(){ return invParams;}
  
}; // class Eigsolve_Mugiq 



#endif // _EIGSOLVE_MUGIQ_H
