#ifndef _MUGIQ_H
#define _MUGIQ_H

/** 
 * @file mugiq.h
 * @brief Main header file for the MUGIQ library
 *
 */

#include <quda.h>


#ifdef __cplusplus
extern "C" {
#endif
  
  /** Wrapper function that calls the QUDA eigensolver to compute eigenvectors and eigenvalues
   * @param h_evecs  Array of pointers to application eigenvectors
   * @param h_evals  Host side eigenvalues
   * @param param Contains all metadata regarding the type of solve.
   */
  void computeEvecs(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *eigParams);
  
#ifdef __cplusplus
}
#endif


#endif // _MUGIQ_H
