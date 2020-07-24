#ifndef _LOOP_MUGIQ_H
#define _LOOP_MUGIQ_H

#include <quda.h>
#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>

using namespace quda;

template <typename Float>
class Loop_Mugiq {

private:

  struct MugiqTraceParam;
  class MugiqShifts;
  
  MugiqTraceParam *trParams; // Trace Parameter structure

  MugiqShifts *shifts;       // The shifts object
  
  Eigsolve_Mugiq *eigsolve; // The eigsolve object (This class is a friend of Eigsolve_Mugiq)

  //- Data buffers
  complex<Float> *dataPos_d;      //-- Device Position space correlator (local)
  complex<Float> *dataPos_h;      //-- Host Position space correlator (local, needed only when no Momentum-projection is performed)
  complex<Float> *dataMom_d;      //-- Device output buffer of cuBlas (local)
  complex<Float> *dataMom_h;      //-- Host output of cuBlas momentum projection (local)
  complex<Float> *dataMom_gs;     //-- Host Globally summed momentum projection buffer (local)
  complex<Float> *dataMom;        //-- Host Final result (global summed, gathered) of momentum projection

  const size_t SizeCplxFloat = sizeof(complex<Float>);
  
  long long nElemMomTot; // Number of elements in global momentum-space data buffers
  long long nElemMomLoc; // Number of elements in local  momentum-space data buffers
  long long nElemPosLoc; // Number of elements in local  position-space data buffers

  
  /** @brief Allocate host and device data buffers
   */
  void allocateDataMemory();

  /** @brief Free host and device data buffers
   */
  void freeDataMemory();

  
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
struct Loop_Mugiq<Float>::MugiqTraceParam {

  const int Ndata = N_GAMMA_;   // Number of Gamma matrices (currents, =16)
  const int momDim = MOM_DIM_;  // Momenta dimensions (=3)

  int Nmom;                                 // Number of Momenta
  LoopFTSign FTSign;                        // Sign of the Fourier Transform
  std::vector<std::vector<int>> momMatrix;  // Momenta Matrix
  
  int max_depth;                // maximum depth of transverse shift length (for later)

  MuGiqBool doMomProj;          // whether to do Momentum projection, if false then the position-space trace will be saved

  int localL[N_DIM_];           // local dimensions
  int totalL[N_DIM_];           // global dimensions

  int locT;                     // local  time dimension
  int totT;                     // global time dimension
  long long locV4 = 1;          // local spatial volume
  long long locV3 = 1;          // local 3d volume (no time)
  long long totV3 = 1;          // global 3d volume (no time)

  LoopCalcType calcType; // Type of computation that will take place

  MuGiqBool init; // Whether the structure has been initialized
  
  MugiqTraceParam(MugiqLoopParam *loopParams, ColorSpinorField *x) :
    Nmom(loopParams->Nmom),
    FTSign(loopParams->FTSign),
    max_depth(0),
    doMomProj(loopParams->doMomProj),
    localL{0,0,0,0},
    totalL{0,0,0,0},
    locT(0), totT(0),
    locV4(1), locV3(1), totV3(1),
    calcType(loopParams->calcType),
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

    for(int im=0;im<Nmom;im++)
      momMatrix.push_back(loopParams->momMatrix[im]);

    init = MUGIQ_BOOL_TRUE;
  } // constructor

  ~MugiqTraceParam(){
    init = MUGIQ_BOOL_FALSE;
  }

};


template <typename Float>
class Loop_Mugiq<Float>::MugiqShifts{

private:

  /** @brief Create a new Gauge Field (it's different from the one used for the MG environment!)
   */
  void createCudaGaugeField();

  /** @brief Create a new Gauge Field with Extended Halos (taking corners into account)
   */
  void createExtendedCudaGaugeField();

public:

  MugiqShifts();
  ~MugiqShifts();  

  
};



#endif // _LOOP_MUGIQ_H
