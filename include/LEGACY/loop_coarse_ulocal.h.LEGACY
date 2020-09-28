#ifndef _LOOP_COARSE_H
#define _LOOP_COARSE_H

#include <eigsolve_mugiq.h>
#include <mugiq.h>
#include <util_mugiq.h>

template <typename Float>
void assembleCoarseLoop_uLocal_opt(complex<Float> *loop_dev,
				   Eigsolve_Mugiq *eigsolve,
				   const std::vector<ColorSpinorField*> &unitGammaPos,
				   const std::vector<ColorSpinorField*> &unitGammaMom,
				   MugiqLoopParam *loopParams);

template <typename Float>
void assembleCoarseLoop_uLocal_blas(complex<Float> *loop_dev,
				    Eigsolve_Mugiq *eigsolve,
				    complex<Float> gCoeff[][SPINOR_SITE_LEN_*SPINOR_SITE_LEN_],
				    const std::vector<ColorSpinorField*> &unitGammaPos,
				    const std::vector<ColorSpinorField*> &unitGammaMom,
				    MugiqLoopParam *loopParams);


#endif // _LOOP_COARSE_H
