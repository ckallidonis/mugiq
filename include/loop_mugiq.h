#ifndef _LOOP_MUGIQ_H
#define _LOOP_MUGIQ_H

#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>
#include <disp_state.h>

using namespace quda;

template <typename Float>
class Loop_Mugiq {

private:

  struct LoopComputeParam;
  
  LoopComputeParam *cPrm; // Loop computation Parameter structure

  LoopDispState<Float> *dSt;  // structure holding the state of displacements
  
  Eigsolve_Mugiq *eigsolve; // The eigsolve object (This class is a friend of Eigsolve_Mugiq)

  //- Data buffers
  complex<Float> *dataPos_d;      //-- Device Position space correlator (local)
  complex<Float> *dataPos_h;      //-- Host Position space correlator (local, needed only when no Momentum-projection is performed)
  complex<Float> *dataMom_d;      //-- Device output buffer of cuBlas (local)
  complex<Float> *dataMom_h;      //-- Host output of cuBlas momentum projection (local)
  complex<Float> *dataMom_gs;     //-- Host Globally summed momentum projection buffer (local)
  complex<Float> *dataMom;        //-- Host Final result (global summed, gathered) of momentum projection

  complex<Float> *phaseMatrix_d;  //-- Device buffer of the phase matrix
  
  const size_t SizeCplxFloat = sizeof(complex<Float>);
  
  long long nElemMomTot; // Number of elements in global momentum-space data buffers
  long long nElemMomLoc; // Number of elements in local  momentum-space data buffers
  long long nElemPosLoc; // Number of elements in local  position-space data buffers
  long long nElemPhMat;  // Number of elements in phase matrix


  
  /** @brief Prolongate the coarse eigenvectors to fine fields
   */
  void prolongateEvec(ColorSpinorField *fineEvec, ColorSpinorField *coarseEvec);

  
  /** @brief Print the Parameters of the Loop computation
   */
  void printLoopComputeParams();

  /** @brief Allocate host and device data buffers
   */
  void allocateDataMemory();

  /** @brief Free host and device data buffers
   */
  void freeDataMemory();

  /** @brief Wrapper to copy the gamma matrix coefficients to GPU __constant__ memory
   */
  void copyGammaToConstMem();

  /** @brief Create the Phase matrix, needed for Momentum Projection (Fourier Transform)
   */
  void createPhaseMatrix();
  
  
public:

  Loop_Mugiq(MugiqLoopParam *loopParams_, Eigsolve_Mugiq *eigsolve_);
  ~Loop_Mugiq();

  
  /** @brief Print the Loop data in ASCII format (in stdout for now)
   */
  void printData_ASCII();

  
  /** @brief Wrapper to create the coarse part of the loop
   */
  void computeCoarseLoop();
  
}; // class Loop_Mugiq




template <typename Float>
struct Loop_Mugiq<Float>::LoopComputeParam {

  const int Ndata = N_GAMMA_;   // Number of Gamma matrices (currents, =16)
  const int momDim = MOM_DIM_;  // Momenta dimensions (=3)

  int Nmom;                                 // Number of Momenta
  LoopFTSign FTSign;                        // Sign of the Fourier Transform
  int *momMatrix;                           // Momenta Matrix, follows lexicographic order momDim-inside-Nmom
  
  int max_depth;                // maximum depth of transverse shift length (for later)

  MuGiqBool doMomProj;          // whether to do Momentum projection, if false then the position-space trace will be saved
  MuGiqBool doNonLocal;         // whether to compute loop for non-local currents

  int localL[N_DIM_];           // local dimensions
  int totalL[N_DIM_];           // global dimensions

  int locT;                     // local  time dimension
  int totT;                     // global time dimension
  long long locV4 = 1;          // local spatial volume
  long long locV3 = 1;          // local 3d volume (no time)
  long long totV3 = 1;          // global 3d volume (no time)

  LoopCalcType calcType; // Type of computation that will take place
  
  char pathString[MAX_PATH_LEN_];
  int pathLen;

  MuGiqBool init; // Whether the structure has been initialized

  
  LoopComputeParam(MugiqLoopParam *loopParams, ColorSpinorField *x) :
    Nmom(loopParams->Nmom),
    FTSign(loopParams->FTSign),
    max_depth(0),
    doMomProj(loopParams->doMomProj),
    doNonLocal(loopParams->doNonLocal),
    localL{0,0,0,0},
    totalL{0,0,0,0},
    locT(0), totT(0),
    locV4(1), locV3(1), totV3(1),
    calcType(loopParams->calcType),
    pathString("\0"),
    pathLen(0),
    init(MUGIQ_BOOL_FALSE)
  {
    for(int i=0;i<N_DIM_;i++){
      localL[i] = x->X(i);
      totalL[i] = localL[i] * comm_dim(i);
      locV4 *= localL[i];
      if(i<N_DIM_-1){
	locV3 *= localL[i];
	totV3 *= totalL[i];
      }
    }
    locT = localL[N_DIM_-1];
    totT = totalL[N_DIM_-1];

    if(doMomProj){
      momMatrix = static_cast<int*>(calloc(Nmom*momDim, sizeof(int)));
      for(int im=0;im<Nmom;im++)
	for(int id=0;id<momDim;id++)
	  momMatrix[MOM_MATRIX_IDX(id,im)] = loopParams->momMatrix[im][id];
    }
    
    if(doNonLocal){
      strcpy(pathString, loopParams->pathString);
      pathLen = strlen(pathString);
    }
    printfQuda("%s: Loop compute parameters are set\n", __func__);

    init = MUGIQ_BOOL_TRUE;
  } // constructor

  ~LoopComputeParam(){
    init = MUGIQ_BOOL_FALSE;
    if(doMomProj){
      free(momMatrix);
      momMatrix = nullptr;
    }
  }

};


/************************************************************/
//- Forward declarations of functions called within Loop_Mugiq


/** @brief Define the Gamma matrix coefficient structure and copy it to GPU __constant__ memory
 */
template <typename Float>
void copyGammaCoeffStructToSymbol();


/** @brief Create the phase matrix on GPU
 */
template <typename Float>
void createPhaseMatrixGPU(complex<Float> *phaseMatrix_d, const int* momMatrix_h,
                          long long locV3, int Nmom, int FTSign,
			  const int localL[], const int totalL[]);



/** @brief Perform the loop contractions
 */
template <typename Float>
void performLoopContraction(complex<Float> *loopData_d, ColorSpinorField *evecL, ColorSpinorField *evecR);




#endif // _LOOP_MUGIQ_H
