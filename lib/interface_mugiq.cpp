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
#include <loop_mugiq.h>
#include <eigsolve_mugiq.h>
#include <mg_mugiq.h>
#include <util_mugiq.h>
#include <interface_mugiq.h>

#include <type_traits>

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

template <typename Float, QudaFieldOrder fieldOrder>
void computeLoop(MugiqLoopParam loopParams, Eigsolve_Mugiq *eigsolve){

  Loop_Mugiq<Float,fieldOrder> *loop = new Loop_Mugiq<Float,fieldOrder>(&loopParams, eigsolve);
  
  loop->computeCoarseLoop();
  
  if(loopParams.writeMomSpaceHDF5 != MUGIQ_BOOL_FALSE ||
     loopParams.writePosSpaceHDF5 != MUGIQ_BOOL_FALSE)
    loop->writeLoopsHDF5();
  else warningQuda("%s: Will NOT write output data!\n", __func__);

  //- Clean-up
  delete loop;
}

//- Compute disconnected loops, top level function
template <typename Float>
void computeLoop(QudaMultigridParam mgParams, QudaEigParam QudaEigParams, MugiqLoopParam loopParams,
		 MuGiqBool computeCoarse, MuGiqBool useMG){

  QudaInvertParam *invParams = QudaEigParams.invert_param;
  
  pushVerbosity(QudaEigParams.invert_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(invParams);
    printQudaEigParam(&QudaEigParams);
  }

  //- Create the Multigrid environment
  MG_Mugiq *mg_env = nullptr;  
  Eigsolve_Mugiq *eigsolve = nullptr;
  
  //- Create the eigensolver environment
  MugiqEigParam *eigParams = new MugiqEigParam(&QudaEigParams);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);
  if(useMG){
    printfQuda("\n%s: Will compute disconnected loops using Multi-grid deflation!\n", __func__);  
    mg_env = newMG_Mugiq(&mgParams, &QudaEigParams);
    eigsolve = new Eigsolve_Mugiq(eigParams, mg_env, &profileEigensolveMuGiq, computeCoarse);
  }
  else{
    printfQuda("\n%s: Will NOT use Multi-grid deflation to compute disconnected loops!\n", __func__);  
    eigsolve = new Eigsolve_Mugiq(eigParams, &profileEigensolveMuGiq);
  }
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

  eigsolve->printInfo();
  
  //- Compute eigenvectors and (local) eigenvalues
  eigsolve->computeEvecs();
  eigsolve->computeEvals();
  eigsolve->printEvals();
  
  const QudaPrecision ePrec = eigsolve->getEvecs()[0]->Precision();
  const int ePrecInt = static_cast<int>(ePrec);
  if(ePrec == QUDA_SINGLE_PRECISION && typeid(Float) == typeid(float))
    printfQuda("\n%s: Running in single precision!\n", __func__);
  else if(ePrec == QUDA_DOUBLE_PRECISION && typeid(Float) == typeid(double))
    printfQuda("\n%s: Running in double precision!\n", __func__);
  else errorQuda("Missmatch between eigenvector precision %d and templated precision %zu\n", ePrecInt, sizeof(Float));
  
  /** Compute loop, template on Field Order
   *  Currently supported field orders are:
   *  QUDA_FLOAT2_FIELD_ORDER: This is set when running with MG AND using the coarse operator/eigenvectors
   *  QUDA_FLOAT4_FIELD_ORDER: This is set when running with no MG, or with MG and the FINE operator/eigenvectors
   */
  if(eigsolve->getEvecs()[0]->FieldOrder() == QUDA_FLOAT2_FIELD_ORDER){
    if(!(useMG && computeCoarse))
      errorQuda("%s: Got FieldOrder = FLOAT2, but useMGenv = FALSE and computeCoarse = FALSE\n", __func__);
    computeLoop<Float, QUDA_FLOAT2_FIELD_ORDER>(loopParams, eigsolve);
  }
  else if(eigsolve->getEvecs()[0]->FieldOrder() == QUDA_FLOAT4_FIELD_ORDER){
    if(useMG && computeCoarse)
      errorQuda("%s: Got FieldOrder = FLOAT4, but useMGenv = TRUE and computeCoarse = TRUE\n", __func__);
    computeLoop<Float, QUDA_FLOAT4_FIELD_ORDER>(loopParams, eigsolve);
  }

  //- Clean-up
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_FREE);
  delete eigsolve;
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_FREE);
  if(useMG) delete mg_env;
  
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_TOTAL);
  printProfileInfo(profileEigensolveMuGiq);

  popVerbosity();
  saveTuneCache();
}

template void computeLoop<double>(QudaMultigridParam mgParams, QudaEigParam QudaEigParams, MugiqLoopParam loopParams,
				  MuGiqBool computeCoarse, MuGiqBool useMG);
template void computeLoop<float>(QudaMultigridParam mgParams, QudaEigParam QudaEigParams, MugiqLoopParam loopParams,
				 MuGiqBool computeCoarse, MuGiqBool useMG);
