#ifndef _LOOP_COARSE_H
#define _LOOP_COARSE_H

#include <eigsolve_mugiq.h>
#include <mugiq.h>

template <typename Float>
void assembleCoarseLoop_uLocal(complex<Float> *loop_dev,
			       MG_Mugiq *mg_env, Eigsolve_Mugiq *eigsolve,
			       const std::vector<ColorSpinorField*> &unitGammaPos,
			       const std::vector<ColorSpinorField*> &unitGammaMom,
			       QudaInvertParam *invParams, MugiqLoopParam *loopParams);


#endif // _LOOP_COARSE_H
