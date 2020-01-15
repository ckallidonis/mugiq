#ifndef _MUGIQ_H
#define _MUGIQ_H

/** 
 * @file mugiq.h
 * @brief Main header file for the MUGIQ library
 *
 */

#include <quda.h>
#include <enum_mugiq.h>

#define N_SPIN_ 4
#define N_COLOR_ 3
#define N_GAMMA_ 16
#define SPINOR_SITE_LEN_ (N_SPIN_ * N_COLOR_)
#define GAUGE_SITE_LEN_ (N_COLOR_ * N_COLOR_)
#define GAMMA_LEN_ (N_SPIN_ * N_SPIN_)

#ifdef __cplusplus
extern "C" {
#endif
  
  /** Wrapper function that calls the QUDA eigensolver to compute eigenvectors and eigenvalues
   * @param h_evecs  Array of pointers to application eigenvectors
   * @param h_evals  Host side eigenvalues
   * @param param Contains all metadata regarding the type of solve.
   */
  void computeEvecsQudaWrapper(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *eigParams);

  /** MuGiq interface function that computes eigenvectors and eigenvalues of coarse operators using MG
   * @param mgParams  Contains all MG metadata regarding the type of eigensolve.
   * @param eigParams Contains all metadata regarding the type of solve.
   */
  void computeEvecsMuGiq_MG(QudaMultigridParam mgParams, QudaEigParam eigParams);

  /** MuGiq interface function that computes eigenvectors and eigenvalues of the Dirac operator
   * @param eigParams Contains all metadata regarding the type of solve.
   */
  void computeEvecsMuGiq(QudaEigParam eigParams);


  
#ifdef __cplusplus
}
#endif


#endif // _MUGIQ_H
