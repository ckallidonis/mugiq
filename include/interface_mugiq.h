#ifndef _INTERFACE_MUGIQ_H
#define _INTERFACE_MUGIQ_H


using namespace quda;

//- Forward declarations of QUDA-interface functions not declared in MuGiQ's .h files
quda::cudaGaugeField *checkGauge(QudaInvertParam *param);


//- Forward declarations of functions called from functions inside interface_mugiq.cpp
template <typename Float>
void createGammaCoarseVectors_uLocal(std::vector<ColorSpinorField*> &unitGamma,
                                     MG_Mugiq *mg_env, QudaInvertParam *invParams);

template <typename Float>
void createGammaCoeff();

#endif // _INTERFACE_MUGIQ_H
