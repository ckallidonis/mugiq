#ifndef _INTERFACE_MUGIQ_H
#define _INTERFACE_MUGIQ_H

#include <mugiq.h>

using namespace quda;

//- Forward declarations of QUDA-interface functions not declared in MuGiQ's .h files
quda::cudaGaugeField *checkGauge(QudaInvertParam *param);


//- Forward declarations of functions called from functions inside interface_mugiq.cpp
template <typename Float>
void createCoarseLoop_uLocal(complex<Float> *loop_dev, MugiqLoopParam *loopParams,
                             Eigsolve_Mugiq *eigsolve);

#endif // _INTERFACE_MUGIQ_H
