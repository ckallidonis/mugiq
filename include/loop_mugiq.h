#ifndef _LOOP_MUGIQ_H
#define _LOOP_MUGIQ_H

//#include <quda.h>
//#include <multigrid.h>           //- The QUDA MG header file
//#include <color_spinor_field.h>  //- From QUDA
#include <mugiq.h>
#include <eigsolve_mugiq.h>


using namespace quda;


template <typename Float>
class Loop_Mugiq {

private:
  MugiqLoopParam *loopParams; // Parameter structure

  Eigsolve_Mugiq *eigsolve; // The eigsolve object (This class is a friend of Eigsolve_Mugiq)
  
  complex<Float> *data_h; // Loop data on host (CPU)
  complex<Float> *data_d; // Loop data on device (GPU)

  long nElem; // Number of elements in the data buffer

  int globT; // Global time dimension

  size_t loopSize; // Size of data buffers in bytes

  void createCoarseLoop_uLocal_coarseTrace();
  
  
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
