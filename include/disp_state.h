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

  const char *DisplaceFlagArray = "XxYyZzTt" ;
  const char *DisplaceTypeArray[] = {"Covariant"};
  
  const char *DisplaceDirArray[]  = {"x", "y", "z", "t"};
  const char *DisplaceSignArray[] = {"-", "+"};  
  
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


//- Some enums about the Displacements

#define nDisplaceFlags 8
#define nDisplaceTypes 1

typedef enum DisplaceFlag_s {
  DispStr_None = -1,
  DispStr_X = 0,  // +x
  DispStr_x = 1,  // -x
  DispStr_Y = 2,  // +y
  DispStr_y = 3,  // -y
  DispStr_Z = 4,  // +z
  DispStr_z = 5,  // -z
  DispStr_T = 6,  // +t
  DispStr_t = 7,  // -t
} DisplaceFlag;


typedef enum DisplaceDir_s {
  DispDirNone = -1,
  DispDir_x = 0,
  DispDir_y = 1,
  DispDir_z = 2,
  DispDir_t = 3
} DisplaceDir;


typedef enum DisplaceSign_s {
  DispSignNone  = -1,
  DispSignMinus =  0,
  DispSignPlus  =  1
} DisplaceSign;


typedef enum DisplaceType_s {
  InvalidDisplace = -1,
  CovDisplace = 0
} DisplaceType;


#endif
