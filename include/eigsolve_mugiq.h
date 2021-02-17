#ifndef _EIGSOLVE_MUGIQ_H
#define _EIGSOLVE_MUGIQ_H

#include <quda.h>
#include <multigrid.h>           //- The QUDA MG header file
#include <color_spinor_field.h>  //- From QUDA
#include <mg_mugiq.h>
#include <enum_mugiq.h>

using namespace quda;

// Forward declaration of the QUDA-interface function that is needed here
cudaGaugeField *checkGauge(QudaInvertParam *param);


struct MugiqEigParam {
  
  QudaEigParam *QudaEigParams;     // The Quda Eigensolver parameter structure
  
  MuGiqEigOperator diracType = MUGIQ_EIG_OPERATOR_INVALID; // Type of Dirac operator for eigenpair calculation, M, Mdag, MdagM, MMdag
  
  int nEv;    // Number of eigenvectors we want
  int nKr;    // Size of Krylov Space
  double tol; // Tolerance of the eigensolve
  
  // Polynomial acceleration parameters
  MuGiqBool use_poly_acc;
  int poly_acc_deg;
  double a_min;
  double a_max;
  
  // Eig-params constructor for use with MG eigensolves
  MugiqEigParam(QudaEigParam *QudaEigParams_) :
    QudaEigParams(QudaEigParams_),
    diracType(MUGIQ_EIG_OPERATOR_INVALID),
    nEv(QudaEigParams_->nEv),
    nKr(QudaEigParams_->nKr),
    tol(QudaEigParams_->tol),
    use_poly_acc(QudaEigParams->use_poly_acc == QUDA_BOOLEAN_YES ? MUGIQ_BOOL_TRUE : MUGIQ_BOOL_FALSE),
    poly_acc_deg(0),
    a_min(0),
    a_max(0)
  {
    if(use_poly_acc){
      poly_acc_deg = QudaEigParams->poly_deg;
      a_min = QudaEigParams->a_min;
      a_max = QudaEigParams->a_max;
    }    
  } //-constructor
  
};


	
class Eigsolve_Mugiq {

private:

  //- Loop_Mugiq will substantially use members of Eigsolve_Mugiq
  //- therefore it deserves to be a friend of this class
  template <typename Float, QudaFieldOrder fieldOrder>
  friend class Loop_Mugiq;
  
  MugiqEigParam *eigParams;
  
  MuGiqBool eigInit;        // Initialization switch
  MuGiqBool useMGenv;  // Whether to use MG environment operators, etc

  MG_Mugiq *mg_env; // The Multigrid environment

  QudaMultigridParam *mgParams;   // Multigrid parameter structure
  QudaInvertParam *invParams;     // Inverter parameter structure

  TimeProfile *eigProfile; // Used for profiling
  
  const Dirac *dirac;
  DiracMatrix *mat; // The Dirac operator whose eigenpairs we are computing
  DiracMatrix *matFine; // The Dirac operator whose eigenpairs we are computing

  //- This switch is required so that the dirac object is NOT
  //- deleted when NOT created within Eigsolve.
  //- In this case, it will be deleted from the Multi-grid environment
  MuGiqBool diracCreated;
  
  
  std::vector<ColorSpinorField *> eVecs; // Eigenvectors
  std::vector<Complex> *eVals_quda; // Eigenvalues from the Quda eigensolver
  std::vector<Complex> *eVals; // Eigenvalues computed within the Eigsolve_Mugiq class

  // Singular values of the non-Hermitian M operator
  // These are also the eigenvalues of the Hermitian g5*M
  // sigma = sqrt(eVals)
  std::vector<double> *eVals_sigma; 

  std::vector<ColorSpinorField *> tmpCSF; // Temporary field(s)
  
  std::vector<double> *evals_res;

  // Whether we are computing eigenpairs of the coarse Dirac operator
  // (significant only when using Multigrid, default true) 
  MuGiqBool computeCoarse; 

  
public:
  Eigsolve_Mugiq(MugiqEigParam *eigParams_,
		 MG_Mugiq *mg_env_,
		 TimeProfile *eigProfile_,
		 MuGiqBool computeCoarse_ = MUGIQ_BOOL_TRUE);

  Eigsolve_Mugiq(MugiqEigParam *eigParams_, TimeProfile *profile_);
  ~Eigsolve_Mugiq();

  
  /** @brief Allocate memory for the fine-grid eigenvectors
  */
  void allocateFineEvecs();
  
  /** @brief Allocate memory for the coarse-grid eigenvectors
  */
  void allocateCoarseEvecs();
  
  /** @brief Allocate memory for the eigenvalues
  */
  void allocateEvals();
  
  /** @brief Create the Dirac operator
  */
  void createDiracOperator();

  /** @brief Determine the type of Dirac operator for eigenpair calculation
      @param[in] eigParams_: The eigen-solver parameter structure
   */
  MuGiqEigOperator determineEigOperator(QudaEigParam *QudaeigParams_);

  /** @brief Create a New Dirac Matrix, whose eigenpairs will be computed
  */
  void createNewDiracMatrix();
  
  /** @brief Perform basic checks based on parameter structure input values
   */
  void makeChecks();

  /** @brief Print Eigenvolver info
   */
  void printInfo();

  /** @brief Compute eigenvectors of Dirac operator
   */
  void computeEvecs();

  /** @brief Compute eigenvalues
   */
  void computeEvals();
  
  /** @brief Perform the projection: out = \sum_i evecs_i * dot(evecs_i*,\gamma_5 * fine_op * in) / eval_i
   */
  void projectVector(ColorSpinorField &out, ColorSpinorField &in);

  /** @brief Compute eigenvalues
   */
  void printEvals();

  /** @brief Accessor to get approx. right singular vector of the given fine/coarse operator
   */
  std::vector<ColorSpinorField *> &getEvecs(){ return eVecs;}
  
  /** @brief Accessor to get the approx. eigenvalues of \gamma_5 * the given fine/coarse operator;
   *         that is, evals[i] = dot(evecs[i], \gamma_5 * fine/coarse operator * evecs[i])
   */
  std::vector<Complex> *getEvals(){ return eVals;}
  
  /** @brief Accessor to get the Multigrid environment structure
   */
  MG_Mugiq* getMGEnv(){ return mg_env;}
  
}; // class Eigsolve_Mugiq 



#endif // _EIGSOLVE_MUGIQ_H
