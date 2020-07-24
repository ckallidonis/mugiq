#ifndef _DISP_STATE_H
#define _DISP_STATE_H

#include <quda.h>
#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>


using namespace quda;


template <typename T>
class LoopDispState {

private:

  template <typename Float>
  friend class Loop_Mugiq;

  //- The pointer with the gauge data coming from the interface
  void *gaugePtr[N_DIM_];

  //- The extended gauge field that will be used for covariant derivative (displacement) of color-spinor fields
  cudaGaugeField *gaugeField; 

  //- Auxilliary color-spinor-field used for shifts
  cudaColorSpinorField *auxCSF;

  
  /** @brief Create a new Gauge Field (it's different from the one used for the MG environment!)
   */
  void createCudaGaugeField();
  
  /** @brief Create a new Gauge Field with Extended Halos (taking corners into account)
   */
  void createExtendedCudaGaugeField();

public:

  LoopDispState(MugiqLoopParam *loopParams_);
  ~LoopDispState();


};


#endif
