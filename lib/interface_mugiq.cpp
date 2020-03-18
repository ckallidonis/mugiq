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
#include <linalg_mugiq.h>
#include <eigsolve_mugiq.h>
#include <mg_mugiq.h>
#include <util_mugiq.h>
#include <loop_coarse.h>
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

void computeEvecsQudaWrapper(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *eigParams){
  
  // Call the QUDA function
  printfQuda("\n%s: Calling the QUDA-library function for computing eigenvectors!\n", __func__);
  eigensolveQuda(eVecs_host, eVals_host, eigParams);
}


//- Create the Multigrid environment
MG_Mugiq* newMG_Mugiq(QudaMultigridParam *mgParams, QudaEigParam *eigParams) {

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
void computeEvecsMuGiq_MG(QudaMultigridParam mgParams, QudaEigParam eigParams){

  printfQuda("\n%s: Using MuGiq interface to compute eigenvectors of coarse Operator using MG!\n", __func__);

  pushVerbosity(eigParams.invert_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(eigParams.invert_param);
    printQudaEigParam(&eigParams);
  }  

  //- Create the Multigrid environment
  MG_Mugiq *mg_env = newMG_Mugiq(&mgParams, &eigParams);
  
  //- Create the eigensolver environment
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);  
  Eigsolve_Mugiq *eigsolve = new Eigsolve_Mugiq(mg_env, &eigParams, &profileEigensolveMuGiq);
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

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
void computeEvecsMuGiq(QudaEigParam eigParams){

  printfQuda("\n%s: Using MuGiq interface to compute eigenvectors of Dirac operator!\n", __func__);

  pushVerbosity(eigParams.invert_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(eigParams.invert_param);
    printQudaEigParam(&eigParams);
  }  
  
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  
  //- Create the eigensolver environment
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);  
  Eigsolve_Mugiq *eigsolve = new Eigsolve_Mugiq(&eigParams, &profileEigensolveMuGiq);
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

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


//- Compute ultra-local (no shifts) disconnected loops using Multigrid deflation
void computeLoop_uLocal_MG(QudaMultigridParam mgParams, QudaEigParam eigParams, MugiqLoopParam loopParams){

  printfQuda("\n%s: Will compute disconnected loops using Multi-grid deflation!\n", __func__);  

  QudaInvertParam *invParams = eigParams.invert_param;
  
  pushVerbosity(eigParams.invert_param->verbosity);
  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
    printQudaInvertParam(invParams);
    printQudaEigParam(&eigParams);
  }

  //- Create the Multigrid environment
  MG_Mugiq *mg_env = newMG_Mugiq(&mgParams, &eigParams);

  //- Create the eigensolver environment
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_TOTAL);
  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_INIT);
  Eigsolve_Mugiq *eigsolve = new Eigsolve_Mugiq(mg_env, &eigParams, &profileEigensolveMuGiq);
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_INIT);

  //- Compute eigenvectors and (local) eigenvalues
  eigsolve->computeEvecs();
  eigsolve->computeEvals();
  eigsolve->printEvals();

  
  //- Assemble the coarse part of the loop
  void *loop_h = nullptr;

  int globT = mg_env->mg_solver->B[0]->X(3) * comm_dim(3); //- Global time dimension
  long loopElem = loopParams.Nmom * globT * N_GAMMA_;

  QudaPrecision ePrec = eigsolve->getEvecs()[0]->Precision();
  if(ePrec == QUDA_DOUBLE_PRECISION){
    size_t loopSize = sizeof(complex<double>) * loopElem;
    loop_h = static_cast<complex<double>*>(malloc(loopSize));
    if(loop_h == NULL) errorQuda("%s: Could not allocate host loop buffer for precision %d\n", __func__, ePrec);
    memset(loop_h, 0, loopSize);

    createCoarseLoop_uLocal<double>(static_cast<complex<double>*>(loop_h),
				    mg_env, eigsolve,
				    invParams, &loopParams);
  }
  else if(ePrec == QUDA_SINGLE_PRECISION){
    size_t loopSize = sizeof(complex<float>) * loopElem;
    loop_h = static_cast<complex<float>*>(malloc(loopSize));
    if(loop_h == NULL) errorQuda("%s: Could not allocate host loop buffer for precision %d\n", __func__, ePrec);
    memset(loop_h, 0, loopSize);

    createCoarseLoop_uLocal<float>(static_cast<complex<float>*>(loop_h),
				   mg_env, eigsolve,
				   invParams, &loopParams);
  }
  //-----------------------------------------------------------


  if(loopParams.printASCII == MUGIQ_BOOL_TRUE){

  }
  
  
  //- Clean-up
  free(loop_h);
  loop_h = nullptr;

  profileEigensolveMuGiq.TPSTART(QUDA_PROFILE_FREE);
  delete eigsolve;
  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_FREE);
  delete mg_env;

  profileEigensolveMuGiq.TPSTOP(QUDA_PROFILE_TOTAL);
  printProfileInfo(profileEigensolveMuGiq);

  popVerbosity();
  saveTuneCache();
}

  


/** Deprecated code ***
std::vector<ColorSpinorField *> tmpCSF;
ColorSpinorParam ucsParam(*(mg_env->mg_solver->B[0]));
ucsParam.create = QUDA_ZERO_FIELD_CREATE;
ucsParam.location = QUDA_CUDA_FIELD_LOCATION;
QudaPrecision coarsePrec = eigParams.invert_param->cuda_prec;
ucsParam.setPrecision(coarsePrec);
int nCoarseLevels = mgParams.n_level-1;
int nextCoarse = nCoarseLevels - 1;

//-Create coarse fields and allocate coarse eigenvectors recursively
tmpCSF.push_back(ColorSpinorField::Create(ucsParam)); //- tmpCSF[0] is a fine field
for(int lev=0;lev<nextCoarse;lev++){
  tmpCSF.push_back(tmpCSF[lev]->CreateCoarse(mgParams.geo_block_size[lev],
					     mgParams.spin_block_size[lev],
					     mgParams.n_vec[lev],
					     coarsePrec,
					     mgParams.setup_location[lev+1]));
 }//-lev

for(int i=0;i<nUnit;i++){
  unitGammaPos.push_back(tmpCSF[nextCoarse]->CreateCoarse(mgParams.geo_block_size[nextCoarse],
						       mgParams.spin_block_size[nextCoarse],
						       mgParams.n_vec[nextCoarse],
						       coarsePrec,
						       mgParams.setup_location[nextCoarse+1]));
 }
*/
