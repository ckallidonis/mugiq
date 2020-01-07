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
//#include <multigrid.h>

// MUGIQ header files
#include <mugiq.h>
#include <mugiq_internal.h>
#include <linalg_mugiq.h>
#include <mg_mugiq.h>
#include <eigsolve_mugiq.h>

//- Profiling
static quda::TimeProfile profileEigensolveMuGiq("computeEvecsMuGiq");
static quda::TimeProfile profileMuGiqMG("newMG_Mugiq");

using namespace quda;

static void printProfileInfo(TimeProfile profile){
  if (getVerbosity() >= QUDA_SUMMARIZE){
    printfQuda("\nPROFILE_INFO:\n");
    profile.Print();
  }
}

//- Interface functions begin here

void computeEvecsQudaWrapper(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *eigParams){
  
  // Call the QUDA function
  printfQuda("\n%s: Calling the QUDA-library function for computing eigenvectors!\n", __func__);
  eigensolveQuda(eVecs_host, eVals_host, eigParams);
}


//- This function is similar to the QUDA eigensolveQuda() function, but adjusted according to MuGiq needs
void computeEvecsMuGiq(QudaEigParam *eigParams){

  printfQuda("\n%s: Using MuGiq interface to compute eigenvectors!\n", __func__);

  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);

  QudaInvertParam *inv_param = eigParams->invert_param;

  if (inv_param->dslash_type != QUDA_WILSON_DSLASH && inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH)
    errorQuda("%s: Supports only Wilson and Wilson-Clover operators for now!\n", __func__);

  // No polynomial acceleration on non-symmetric matrices
  if (eigParams->use_poly_acc && !eigParams->use_norm_op && !(inv_param->dslash_type == QUDA_LAPLACE_DSLASH))
    errorQuda("%s: Polynomial acceleration with non-symmetric matrices not supported", __func__);
  
  pushVerbosity(inv_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(inv_param);
    printQudaEigParam(eigParams);
  }

  cudaGaugeField *cudaGauge = checkGauge(inv_param);

  //- Whether we are running for the "full" or even-odd preconditioned Operator
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE) ||
    (inv_param->solve_type == QUDA_NORMOP_PC_SOLVE) ||
    (inv_param->solve_type == QUDA_NORMERR_PC_SOLVE);

  
  // Create Dirac Operator
  Dirac *d = nullptr;
  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc_solve);
  d = Dirac::create(diracParam);
  
  DiracMatrix *m;
  if (!eigParams->use_norm_op && !eigParams->use_dagger)     m = new DiracM(*d);
  else if (!eigParams->use_norm_op && eigParams->use_dagger) m = new DiracMdag(*d);
  else if (eigParams->use_norm_op && !eigParams->use_dagger) m = new DiracMdagM(*d);
  else if (eigParams->use_norm_op && eigParams->use_dagger)  m = new DiracMMdag(*d);
  //----------------------

  
  //- Create (cuda)ColorSpinorFields to hold the eigenvectors
  //- NULL is used in cpuRaram because the fields are initialized to zero, and not copied to GPU from some host buffer
  const int *X = cudaGauge->X(); //- The Lattice Size
  ColorSpinorParam cpuParam(NULL, *inv_param, X, inv_param->solution_type, inv_param->input_location);
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(eigParams->cuda_prec_ritz, eigParams->cuda_prec_ritz, true);

  std::vector<ColorSpinorField *> eVecs;
  for (int i = 0; i < eigParams->nConv; i++) { eVecs.push_back(ColorSpinorField::Create(cudaParam)); }

  //- These are the eigenvalues coming from the Quda eigensolver
  std::vector<Complex> eVals(eigParams->nConv, 0.0);

  //- Eigenvalues computed from MuGiq, given the Quda eigensolver eigenvectors
  std::vector<Complex> eVals_mugiq(eigParams->nConv, 0.0);

  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

  
  //- Perform eigensolve
  EigenSolver *eig_solve = EigenSolver::create(eigParams, *m, profileEigensolveMuGiq);
  (*eig_solve)(eVecs, eVals);

  computeEvalsMuGiq(*m, eVecs, eVals_mugiq, eigParams);
  
  //- Clean up
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_FREE);
  delete eig_solve;
  delete m;
  delete d;
  for (int i = 0; i < eigParams->nConv; i++) delete eVecs[i];
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_FREE);
  
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_TOTAL);
  printProfileInfo(profileEigensolveMuGiq);

  popVerbosity();
  saveTuneCache();
}

MG_Mugiq* newMG_Mugiq(QudaMultigridParam *mgParams, QudaEigParam *eigParams) {

  pushVerbosity(mgParams->invert_param->verbosity);

  profileMuGiqMG.TPSTART(QUDA_PROFILE_TOTAL);
  MG_Mugiq *mg = new MG_Mugiq(mgParams, eigParams, profileMuGiqMG);
  profileMuGiqMG.TPSTOP(QUDA_PROFILE_TOTAL);

  printProfileInfo(profileMuGiqMG);
  saveTuneCache();
  popVerbosity();

  return mg;
}

//- The purpose of this function is to compute the eigenvalues and eigenvectors of the coarse Dirac operator using MG
void computeEvecsMuGiq_MG(QudaMultigridParam mgParams, QudaEigParam eigParams){

  printfQuda("\n%s: Using MuGiq interface to compute eigenvectors of coarse Operator using MG!\n", __func__);

  //- Create the Multigrid environment
  MG_Mugiq *mg_mugiq = newMG_Mugiq(&mgParams, &eigParams);

  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);

  //- Create the eigensolver environment
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);  
  Eigsolve_Mugiq *eigsolve = new Eigsolve_Mugiq(mg_mugiq, &eigParams, profileEigensolveMuGiq);
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);


  //- Clean-up
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_FREE);
  delete eigsolve;
  delete mg_mugiq;
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_FREE);

  
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_TOTAL);
  printProfileInfo(profileEigensolveMuGiq);
}
