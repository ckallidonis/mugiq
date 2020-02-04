#ifndef _INTERFACE_MUGIQ_H
#define _INTERFACE_MUGIQ_H

#include <mugiq.h>

using namespace quda;

//- Forward declarations of QUDA-interface functions not declared in MuGiQ's .h files
quda::cudaGaugeField *checkGauge(QudaInvertParam *param);


//- Forward declarations of functions called from functions inside interface_mugiq.cpp
template <typename Float>
void createGammaCoarseVectors_uLocal(std::vector<ColorSpinorField*> &unitGammaPos,
				     std::vector<ColorSpinorField*> &unitGammaMom,
                                     MG_Mugiq *mg_env, QudaInvertParam *invParams,
				     MugiqLoopParam *loopParams);

template <typename Float>
void createGammaCoeff();

#endif // _INTERFACE_MUGIQ_H
