#ifndef _LINALG_MUGIQ_H
#define _LINALG_MUGIQ_H

/** 
 * @file linalg_mugiq.h
 * @brief Header file containing function declarations related to linear algebra
 *
 */

/*
#include <vector>
#include <quda.h>
#include <dirac_quda.h>
#include <color_spinor_field.h>
*/

#include <complex.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <color_spinor_field.h>

using namespace quda;

#ifdef __cplusplus
extern "C" {
#endif

  
  /** @brief Compute the eigenvalues
   * @param[in]  M         Dirac operator whose eigenvalues are calculated
   * @param[in]  v         The eigenvectors
   * @param[out] lambda    The computd eigenvalues
   * @param[in]  eigParams The eigensolver parameter structure
   */  
  void computeEvalsMuGiq(const DiracMatrix &mat, std::vector<ColorSpinorField *> &v,
			 std::vector<Complex> &lambda, QudaEigParam *eigParams);
  
  
#ifdef __cplusplus
}
#endif

#endif // _LINALG_MUGIQ_H
