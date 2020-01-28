#ifndef _LOOP_COARSE_H
#define _LOOP_COARSE_H

#include <eigsolve_mugiq.h>

template <typename Float>
void assembleLoopCoarsePart_uLocal(complex<Float> *loop_h,
				   Eigsolve_Mugiq *eigsolve,
				   const std::vector<ColorSpinorField*> &unitGamma);


#endif // _LOOP_COARSE_H
