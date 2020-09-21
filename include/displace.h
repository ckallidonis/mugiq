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
  DispFlagNone = MUGIQ_INVALID_ENUM,
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
  DispDirNone = MUGIQ_INVALID_ENUM,
  DispDir_x = 0,
  DispDir_y = 1,
  DispDir_z = 2,
  DispDir_t = 3
} DisplaceDir;


typedef enum DisplaceSign_s {
  DispSignNone  = MUGIQ_INVALID_ENUM,
  DispSignMinus =  0,
  DispSignPlus  =  1
} DisplaceSign;


typedef enum DisplaceType_s {
  InvalidDisplace = MUGIQ_INVALID_ENUM,
  CovDisplace = 0
} DisplaceType;


template <typename T>
class Displace {

private:

  template <typename Float>
  friend class Loop_Mugiq;
  
  const std::vector<std::string> DisplaceFlagArray {"+x","-x","+y","-y","+z","-z","+t","-t"} ;
  const char *DisplaceTypeArray[N_DISPLACE_TYPES] = {"Covariant"};
  
  const char *DisplaceDirArray[N_DIM_]  = {"x", "y", "z", "t"};
  const char *DisplaceSignArray[N_DISPLACE_SIGNS] = {"-", "+"};  
  
  std::string  dispString;    //- Current Displacement string
  char dispString_c[3];       //- Current Displacement string in char type
  DisplaceFlag dispFlag;      //- Enum of displacement string (helpful for switch cases)
  DisplaceDir  dispDir;       //- Direction of displacement
  DisplaceSign dispSign;      //- Sign of displacement
  
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
  void setupDisplacement(std::string dStr);

  /** @brief Parse the displacement string to get the displacement flag enum
   */
  DisplaceFlag WhichDisplaceFlag();

  /** @brief Get the displacement direction from the displacement flag
   */
  DisplaceDir WhichDisplaceDir();

  /** @brief Get the displacement sign from the displacement flag
   */
  DisplaceSign WhichDisplaceSign();

  

  
public:

  Displace(MugiqLoopParam *loopParams_);
  ~Displace();


};




#endif // _DISPLACE_H
