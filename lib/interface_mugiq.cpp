#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <complex.h>

#include <quda.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <eigensolve_quda.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <algorithm>
#include <random_quda.h>
#include <mpi_comm_handle.h>


// MUGIQ header files
#include <mugiq.h>

//- Forward declarations of QUDA-interface functions not declared in the .h files, and are called here
namespace quda{
  void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve);
}
quda::cudaGaugeField *checkGauge(QudaInvertParam *param);


using namespace quda;


//- Profiling
static TimeProfile profEigsolve("computeEvecsMuGiq");




void computeEvecsQudaWrapper(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *eigParams){

  printfQuda("%s: Calling the QUDA-library function for computing eigenvectors!\n", __func__);
  
  // Call the QUDA function
  eigensolveQuda(eVecs_host, eVals_host, eigParams);
}


//- This function is similar to the QUDA eigensolveQuda() function, but adjusted according to MuGiq needs
void computeEvecsMuGiq(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *eigParams){

  printfQuda("%s: Using MuGiq interface to compute eigenvectors!\n", __func__);

  profEigsolve.TPSTART(QUDA_PROFILE_TOTAL);
  profEigsolve.TPSTART(QUDA_PROFILE_INIT);

  QudaInvertParam *inv_param = eigParams->invert_param;

  if (inv_param->dslash_type != QUDA_WILSON_DSLASH && inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH)
    errorQuda("%s: Supports only Wilson and Wilson-Clover operators for now!\n", __func__);

  //  if (!initialized) errorQuda("QUDA not initialized");

  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(inv_param);
    printQudaEigParam(eigParams);
  }

  cudaGaugeField *cudaGauge = checkGauge(inv_param);

  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) || (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE)
    || (inv_param->solve_type == QUDA_NORMERR_PC_SOLVE);

  
  // Create Dirac Operator
  Dirac *d = nullptr;
  Dirac *dSloppy = nullptr;
  Dirac *dPre = nullptr;

  createDirac(d, dSloppy, dPre, *inv_param, pc_solve);
  Dirac &dirac = *d;
  //----------

  const int *X = cudaGauge->X(); //- The Lattice Size

  // create wrappers around application vector set
  ColorSpinorParam cpuParam(eVecs_host[0], *inv_param, X, inv_param->solution_type, inv_param->input_location);
  std::vector<ColorSpinorField *> eVecs_host_;
  for (int i = 0; i < eigParams->nConv; i++) {
    cpuParam.v = eVecs_host[i];
    eVecs_host_.push_back(ColorSpinorField::Create(cpuParam));
  }

  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(eigParams->cuda_prec_ritz, eigParams->cuda_prec_ritz, true);

  std::vector<Complex> evals(eigParams->nConv, 0.0);
  std::vector<ColorSpinorField *> kSpace;
  for (int i = 0; i < eigParams->nConv; i++) { kSpace.push_back(ColorSpinorField::Create(cudaParam)); }

  // No polynomial acceleration on non-symmetric matrices
  if (eigParams->use_poly_acc && !eigParams->use_norm_op && !(inv_param->dslash_type == QUDA_LAPLACE_DSLASH))
      errorQuda("Polynomial acceleration with non-symmetric matrices not supported");

  profEigsolve.TPSTOP(QUDA_PROFILE_INIT);

  DiracMatrix *m;
  if (!eigParams->use_norm_op && !eigParams->use_dagger)     m = new DiracM(dirac);
  else if (!eigParams->use_norm_op && eigParams->use_dagger) m = new DiracMdag(dirac);
  else if (eigParams->use_norm_op && !eigParams->use_dagger) m = new DiracMdagM(dirac);
  else if (eigParams->use_norm_op && eigParams->use_dagger)  m = new DiracMMdag(dirac);
  
  //- Perform eigensolve
  EigenSolver *eig_solve = EigenSolver::create(eigParams, *m, profEigsolve);
  (*eig_solve)(kSpace, evals);

  
  // Copy eigen values back
  for (int i = 0; i < eigParams->nConv; i++) { eVals_host[i] = real(evals[i]) + imag(evals[i]) * _Complex_I; }

  // Transfer Eigenpairs back to host if using GPU eigensolver
  if (!(eigParams->arpack_check)) {
    profEigsolve.TPSTART(QUDA_PROFILE_D2H);
    for (int i = 0; i < eigParams->nConv; i++) *eVecs_host_[i] = *kSpace[i];
    profEigsolve.TPSTOP(QUDA_PROFILE_D2H);
  }

  profEigsolve.TPSTART(QUDA_PROFILE_FREE);
  for (int i = 0; i < eigParams->nConv; i++) delete eVecs_host_[i];
  delete eig_solve;
  delete m;
  delete d;
  delete dSloppy;
  delete dPre;
  for (int i = 0; i < eigParams->nConv; i++) delete kSpace[i];
  profEigsolve.TPSTOP(QUDA_PROFILE_FREE);

  popVerbosity();

  // cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache();

  profEigsolve.TPSTOP(QUDA_PROFILE_TOTAL);
}


