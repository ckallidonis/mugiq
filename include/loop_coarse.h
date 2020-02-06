#ifndef _LOOP_COARSE_H
#define _LOOP_COARSE_H

#include <eigsolve_mugiq.h>
#include <mugiq.h>

template <typename Float>
void assembleLoopCoarsePart_uLocal(complex<Float> *loop_dev,
				   Eigsolve_Mugiq *eigsolve,
				   const std::vector<ColorSpinorField*> &unitGammaPos,
				   const std::vector<ColorSpinorField*> &unitGammaMom,
				   MugiqLoopParam loopParams);


#endif // _LOOP_COARSE_H
