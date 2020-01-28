#ifndef _INTERFACE_MUGIQ_H
#define _INTERFACE_MUGIQ_H


//- Forward declarations of functions called from functions inside interface_mugiq.cpp

template <typename Float>
void createGammaCoarseVectors_uLocal(std::vector<ColorSpinorField*> &unitGamma,
                                     MG_Mugiq *mg_env, QudaInvertParam *invParams);

template <typename Float>
void createGammaCoeff();

template <typename Float>
void assembleLoopCoarsePart_uLocal(Eigsolve_Mugiq *eigsolve, const std::vector<ColorSpinorField*> &unitGamma);

#endif // _INTERFACE_MUGIQ_H
