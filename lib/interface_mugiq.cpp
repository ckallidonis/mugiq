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
//#include <linalg_mugiq.h>
#include <loop_mugiq.h>
#include <eigsolve_mugiq.h>
#include <mg_mugiq.h>
#include <util_mugiq.h>
#include <interface_mugiq.h>

//- Profiling
static quda::TimeProfile profileEigensolveMuGiq("computeEvecsMuGiq");
static quda::TimeProfile profileMuGiqMG("MugiqMG-Init");

using namespace quda;

static void printProfileInfo(TimeProfile profile){
  if (getVerbosity() >= QUDA_SUMMARIZE){
    printfQuda("\nPROFILE_INFO:\n");
    profile.Print();
  }
}

//--------------------------------
//- Interface functions begin here
//--------------------------------

void computeEvecsQudaWrapper(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *QudaEigParams){
  
  // Call the QUDA function
  printfQuda("\n%s: Calling the QUDA-library function for computing eigenvectors!\n", __func__);
  eigensolveQuda(eVecs_host, eVals_host, QudaEigParams);
}


//- Create the Multigrid environment
MG_Mugiq* newMG_Mugiq(QudaMultigridParam *mgParams, QudaEigParam *QudaEigParams) {

  pushVerbosity(mgParams->invert_param->verbosity);

  profileMuGiqMG.TPSTART(QUDA_PROFILE_TOTAL);
  MG_Mugiq *mg = new MG_Mugiq(mgParams, profileMuGiqMG);
  profileMuGiqMG.TPSTOP(QUDA_PROFILE_TOTAL);

  printProfileInfo(profileMuGiqMG);
  popVerbosity();
  saveTuneCache();

  return mg;
}


//- Compute the eigenvalues and eigenvectors of the coarse Dirac operator using MG
void computeEvecsMuGiq_MG(QudaMultigridParam mgParams, QudaEigParam QudaEigParams){

  printfQuda("\n%s: Using MuGiq interface to compute eigenvectors of coarse Operator using MG!\n", __func__);

  pushVerbosity(QudaEigParams.invert_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(QudaEigParams.invert_param);
    printQudaEigParam(&QudaEigParams);
  }  

  //- Create the Multigrid environment
  MG_Mugiq *mg_env = newMG_Mugiq(&mgParams, &QudaEigParams);

  
  //- Create the eigensolver environment
  MugiqEigParam *eigParams = new MugiqEigParam(&QudaEigParams);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);  
  Eigsolve_Mugiq *eigsolve = new Eigsolve_Mugiq(eigParams, mg_env, &profileEigensolveMuGiq);
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

  eigsolve->printInfo();

  //- Compute eigenvectors and (local) eigenvalues
  eigsolve->computeEvecs();
  eigsolve->computeEvals();
  eigsolve->printEvals();

  //- Clean-up
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_FREE);
  delete eigsolve;
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_FREE);
  delete mg_env;
  
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_TOTAL);
  printProfileInfo(profileEigensolveMuGiq);

  popVerbosity();
  saveTuneCache();
}


//- Compute the eigenvalues and eigenvectors of the Dirac operator
void computeEvecsMuGiq(QudaEigParam QudaEigParams){

  printfQuda("\n%s: Using MuGiq interface to compute eigenvectors of Dirac operator!\n", __func__);

  pushVerbosity(QudaEigParams.invert_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(QudaEigParams.invert_param);
    printQudaEigParam(&QudaEigParams);
  }  
  
  //- Create the eigensolver environment
  MugiqEigParam *eigParams = new MugiqEigParam(&QudaEigParams);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);  
  Eigsolve_Mugiq *eigsolve = new Eigsolve_Mugiq(eigParams, &profileEigensolveMuGiq);
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

  eigsolve->printInfo();
  
  //- Compute eigenvectors and (local) eigenvalues
  eigsolve->computeEvecs();
  eigsolve->computeEvals();
  eigsolve->printEvals();
  
  //- Clean-up
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_FREE);
  delete eigsolve;
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_FREE);
  
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_TOTAL);
  printProfileInfo(profileEigensolveMuGiq);

  popVerbosity();
  saveTuneCache();
}


//- Compute disconnected loops using Multigrid deflation
void computeLoop_MG(QudaMultigridParam mgParams, QudaEigParam QudaEigParams, MugiqLoopParam loopParams){

  printfQuda("\n%s: Will compute disconnected loops using Multi-grid deflation!\n", __func__);  

  QudaInvertParam *invParams = QudaEigParams.invert_param;
  
  pushVerbosity(QudaEigParams.invert_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(invParams);
    printQudaEigParam(&QudaEigParams);
  }

  //- Create the Multigrid environment
  MG_Mugiq *mg_env = newMG_Mugiq(&mgParams, &QudaEigParams);

  //- Create the eigensolver environment
  MugiqEigParam *eigParams = new MugiqEigParam(&QudaEigParams);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);
  Eigsolve_Mugiq *eigsolve = new Eigsolve_Mugiq(eigParams, mg_env, &profileEigensolveMuGiq);
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

  eigsolve->printInfo();
  
  //- Compute eigenvectors and (local) eigenvalues
  eigsolve->computeEvecs();
  eigsolve->computeEvals();
  eigsolve->printEvals();

  QudaPrecision ePrec = eigsolve->getEvecs()[0]->Precision();
  if(ePrec == QUDA_SINGLE_PRECISION)
    printfQuda("\n%s: Running in single precision\n", __func__);
  else if(ePrec == QUDA_DOUBLE_PRECISION)
    printfQuda("\n%s: Running in double precision\n", __func__);

  
  //- Determine precision of the calculation
#if (ePrec == QUDA_SINGLE_PRECISION)
  typedef float F;
#elif (ePrec == QUDA_DOUBLE_PRECISION)
  typedef double F;
#endif

  //- Create a new loop object
  Loop_Mugiq<F> *loop = new Loop_Mugiq<F>(&loopParams, eigsolve);

  loop->computeCoarseLoop();

  if(loopParams.printASCII == MUGIQ_BOOL_TRUE) loop->printData_ASCII();
  
  //- Clean-up
#if 0
  free(loop_h);
  loop_h = nullptr;
#endif
  
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_FREE);
  delete eigsolve;
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_FREE);
  delete mg_env;

  delete loop;
  
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_TOTAL);
  printProfileInfo(profileEigensolveMuGiq);

  popVerbosity();
  saveTuneCache();
}
