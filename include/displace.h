#ifndef _DISPLACE_H
#define _DISPLACE_H

#include <quda.h>
#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>
#include <gauge_field.h>

using namespace quda;

#define N_DISPLACE_FLAGS 8
#define N_DISPLACE_TYPES 1
#define N_DISPLACE_SIGNS 2


//- Some enums about the Displacements

typedef enum DisplaceFlag_s {
  DispFlag_None = -1,
  DispFlag_X = 0,  // +x
  DispFlag_x = 1,  // -x
  DispFlag_Y = 2,  // +y
  DispFlag_y = 3,  // -y
  DispFlag_Z = 4,  // +z
  DispFlag_z = 5,  // -z
  DispFlag_T = 6,  // +t
  DispFlag_t = 7,  // -t
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



template <typename T>
class Displace {

private:

  template <typename Float>
  friend class Loop_Mugiq;

  const char *DisplaceFlagArray = "XxYyZzTt" ;
  const char *DisplaceTypeArray[N_DISPLACE_TYPES] = {"Covariant"};
  
  const char *DisplaceDirArray[N_DIM_]  = {"x", "y", "z", "t"};
  const char *DisplaceSignArray[N_DISPLACE_SIGNS] = {"-", "+"};  

  char dispStr;     //- Current Displacement string
  char prevDispStr; //- Previous displacement string 

  DisplaceFlag dispFlag;
  DisplaceDir  dispDir;
  DisplaceSign dispSign;

  
  //- The pointer with the gauge data coming from the interface
  void *gaugePtr[N_DIM_];

  QudaGaugeParam *qGaugePrm;
  
  //- The extended gauge field that will be used for covariant derivative (displacement) of color-spinor fields
  cudaGaugeField *gaugeField; 

  //- Auxilliary color-spinor-field used for displacements
  cudaColorSpinorField *dispVec;

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

  
  /** @brief Set up the displacement
   */
  void setupDisplacement(char dStr);

  /** @brief Parse the displacement flag
   */
  DisplaceFlag ParseDisplaceFlag();

  /** @brief Parse the displacement directory
   */
  DisplaceDir ParseDisplaceDir();

  /** @brief Parse the displacement sign
   */
  DisplaceSign ParseDisplaceSign();

  

  
public:

  Displace(MugiqLoopParam *loopParams_);
  ~Displace();


};




#endif // _DISPLACE_H
