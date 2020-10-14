#ifndef _MUGIQ_H
#define _MUGIQ_H

/** 
 * @file mugiq.h
 * @brief Main header file for the MUGIQ library
 *
 */

#include <quda.h>
#include <enum_mugiq.h>

#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

  //- Allow five entries on average for each of the eight displace directions, +x,-x,+y,-y,+z,-z,+t,-t
  //- Should be more than enough
  //#define MAX_DISPLACE_ENTRIES 40
  
  /* Structure that holds parameters related to the calculation of
   * disconnected quark loops.
   * Will be extended according to compuation demands
   */
  typedef struct MugiqLoopParam_s {

    int Nmom; //- Number of momenta for Fourier Transform
    std::vector<std::vector<int>> momMatrix; //- 2d-Array/vector holding the momenta values, dimensions [Nmom][3]
    LoopFTSign FTSign;
    LoopCalcType calcType;
    MuGiqBool writeMomSpaceHDF5;
    MuGiqBool writePosSpaceHDF5;
    MuGiqBool doMomProj;
    MuGiqBool doNonLocal;
    std::vector<std::string> disp_entry;
    std::vector<std::string> disp_str;
    std::string fname_mom_h5;
    std::string fname_pos_h5;
    std::vector<int> disp_start;
    std::vector<int> disp_stop;
    void *gauge[4];
    QudaGaugeParam *gauge_param;
    
  } MugiqLoopParam;
  
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

  /** MuGiq interface function that computes disconnected quark loops using Multigrid Deflation
   *  and for ultra-local current insertions
   * @param mgParams  Contains all MG metadata regarding the type of eigensolve.
   * @param eigParams Contains all metadata regarding the type of solve.
   * @param loopParams Contains all metadata regarding the loop calculation
   */
  void computeLoop_MG(QudaMultigridParam mgParams, QudaEigParam eigParams, MugiqLoopParam loopParams);
  
#ifdef __cplusplus
}
#endif


#endif // _MUGIQ_H
