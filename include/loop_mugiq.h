#ifndef _LOOP_MUGIQ_H
#define _LOOP_MUGIQ_H

//#include <quda.h>
//#include <multigrid.h>           //- The QUDA MG header file
//#include <color_spinor_field.h>  //- From QUDA
#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>

using namespace quda;


struct MugiqTraceParam {

  const int Ndata = N_GAMMA_;   // Number of Gamma matrices (currents, =16)
  const int momDim = MOM_DIM_;  // Momenta dimensions (=3)

  int Nmom;                                 // Number of Moment
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
  
  MugiqTraceParam(MugiqLoopParam *loopParams, ColorSpinorField *x) :
    Nmom(loopParams->Nmom),
    FTSign(loopParams->FTSign),
    max_depth(0),
    doMomProj(loopParams->doMomProj),
    localL{0,0,0,0},
    totalL{0,0,0,0},
    locT(0), totT(0),
    locV4(1), locV3(1), totV3(1),
    calcType(loopParams->calcType)
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
  } // constructor

};


template <typename Float>
class Loop_Mugiq {

private:
  MugiqTraceParam *params; // Parameter structure

  Eigsolve_Mugiq *eigsolve; // The eigsolve object (This class is a friend of Eigsolve_Mugiq)
  
  complex<Float> *dataMom_h; // Loop data on host (CPU) in momentum-space
  complex<Float> *dataMom_d; // Loop data on device (GPU) in momentum-space

  long nElemMom; // Number of elements in momentum-space data buffers

  size_t loopSizeMom; // Size of momentum-space data buffers in bytes

  
  /** @brief Compute the coarse part of the loop for ultra-local currents, using 
   * an optimized CUDA kernel
   */
  void createCoarseLoop_uLocal_optKernel();
  
  
public:

  Loop_Mugiq(MugiqLoopParam *loopParams_, Eigsolve_Mugiq *eigsolve_);
  ~Loop_Mugiq();

  
  /** @brief Print the Loop data in ASCII format (in stdout for now)
   */
  void printData_ASCII();

  
  /** @brief Wrapper to create the coarse part of the loop
   */
  void createCoarseLoop_uLocal();


  
}; // class Loop_Mugiq

#endif // _LOOP_MUGIQ_H
