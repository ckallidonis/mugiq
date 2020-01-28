#include <loop_coarse.h>
#include <kernels_mugiq.h>


/* Top-level function, called from the interface
 * This function calls a CUDA kernel to assmeble the disconnected quark loop for ultra-local currents (no displacements).
 * Components are the coarse gamma-matrix unity vectors e_i, the coarse eigenpairs v_i, lambda_i and the gamma matrix coefficients,
 * gcoeff, which are already copied to __constant__ memory.
 * The operation performed is:
 * L_n(x,t) = \sum_{m}^{Nev} lambda_m^{-1} v_m^\dag [\sum_{i,j}^{Ns*Nc} gcoeff(n)_{ij} e_i e_j^\dag] v_m
 */
template void assembleLoopCoarsePart_uLocal<double>(Eigsolve_Mugiq *eigsolve, const std::vector<ColorSpinorField*> &unitGamma);
template void assembleLoopCoarsePart_uLocal<float>(Eigsolve_Mugiq *eigsolve, const std::vector<ColorSpinorField*> &unitGamma);

template <typename Float>
void assembleLoopCoarsePart_uLocal(Eigsolve_Mugiq *eigsolve, const std::vector<ColorSpinorField*> &unitGamma){

  Arg_Loop_uLocal<Float> arg(unitGamma, eigsolve->getEvecs(), eigsolve->getEvals());
  Arg_Loop_uLocal<Float> *arg_dev;
  cudaMalloc((void**)&(arg_dev), sizeof(Arg_Loop_uLocal<Float>));
  checkCudaError();
  cudaMemcpy(arg_dev, &arg, sizeof(Arg_Loop_uLocal<Float>), cudaMemcpyHostToDevice);
  checkCudaError();

  if(arg.nParity != 2) errorQuda("%s: This function supports only Full Site Subset fields!\n", __func__);

  //- Clean-up
  cudaFree(arg_dev);
  arg_dev = NULL;

  printfQuda("%s: Ultra-local disconnected loop assembly completed\n", __func__);
}
//-------------------------------------------------------------------------------
