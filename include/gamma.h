#include <kernels_mugiq.h>
#include <utility_kernels.h>
#include <mg_mugiq.h>

void restrictGammaUnitVectors(std::vector<ColorSpinorField*> &unitGamma,
			      std::vector<ColorSpinorField*> &gammaGens,
			      std::vector<ColorSpinorField*> &tmpCSF,
			      MG_Mugiq *mg_env);

template <typename Float>
void createUnphasedGammaUnitVectors(std::vector<ColorSpinorField*> &gammaGens);

template <typename Float>
void createPhasedGammaUnitVectors(std::vector<ColorSpinorField*> &gammaGens,
                                  std::vector<int> mom, LoopFTSign FTsign);

template <typename Float>
void timeDilutePhasedGammaUnitVectors(std::vector<ColorSpinorField*> &gammaGensTD,
				      std::vector<ColorSpinorField*> &gammaGens,
				      int glob_t);

template <typename Float>
void createGammaCoeff(complex<Float> gCoeff[][SPINOR_SITE_LEN_*SPINOR_SITE_LEN_]);
