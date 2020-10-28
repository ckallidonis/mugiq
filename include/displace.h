#ifndef _DISPLACE_H
#define _DISPLACE_H

#include <quda.h>
#include <mugiq.h>
#include <eigsolve_mugiq.h>
#include <util_mugiq.h>
#include <gauge_field.h>

using namespace quda;


template <typename F, QudaFieldOrder order>
class Displace {

private:

  template <typename Float, QudaFieldOrder fieldOrder>
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
  ColorSpinorField *auxDispVec;

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

  /** @brief Reset the displaced Vector to the original, un-displaced eigenvector
   */
  void resetAuxDispVec(ColorSpinorField *fineEvec);

  
  /** @brief Perform the displacement
   */
  void doVectorDisplacement(DisplaceType dispType, ColorSpinorField *displacedEvec, int idisp);

  /** @brief Swap the auxilliary displaced vector with the output displaced vector
   */
  void swapAuxDispVec(ColorSpinorField *displacedEvec);

  

  
public:

  Displace(MugiqLoopParam *loopParams_, ColorSpinorField *csf, QudaPrecision coarsePrec_);
  ~Displace();


};


/****************************************************************/
//- Forward declarations of functions called within Displace class


/** @brief Perform a covariant displacement of the form dst(x) = U_d(x)*src(x+d) - src(x)
 */
template <typename Float>
void performCovariantDisplacementVector(ColorSpinorField *dst, ColorSpinorField *src, cudaGaugeField *gauge,
					DisplaceDir dispDir, DisplaceSign dispSign);

#endif // _DISPLACE_H
