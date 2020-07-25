#ifndef _DISP_STATE_H
#define _DISP_STATE_H

#include <quda.h>
#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>
#include <gauge_field.h>

using namespace quda;


template <typename T>
class LoopDispState {

private:

  template <typename Float>
  friend class Loop_Mugiq;

  //- The pointer with the gauge data coming from the interface
  void *gaugePtr[N_DIM_];

  QudaGaugeParam *qGaugePrm;
  
  //- The extended gauge field that will be used for covariant derivative (displacement) of color-spinor fields
  cudaGaugeField *gaugeField; 

  //- Auxilliary color-spinor-field used for shifts/displacements
  cudaColorSpinorField *auxCSF;

  //- This prevents redundant halo exchange (as set in QUDA)
  static const MuGiqBool redundantComms = MUGIQ_BOOL_FALSE;

  //- Range/Size of extended Halos
  int exRng[N_DIM_];
  
  
  /** @brief Create a new Gauge Field (it's different from the one used for the MG environment!)
   */
  cudaGaugeField *createCudaGaugeField();
  
  /** @brief Create a new Gauge Field with Extended Halos (taking corners into account)
   */
  void createExtendedCudaGaugeField(bool copyGauge=true,
				    bool redundant_comms=redundantComms ? true : false,
				    QudaReconstructType recon=QUDA_RECONSTRUCT_INVALID);

public:

  LoopDispState(MugiqLoopParam *loopParams_);
  ~LoopDispState();


};


#endif
