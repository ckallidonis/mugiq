#include <mg_mugiq.h>
#include <eigsolve_mugiq.h>
#include <interface_mugiq.h>
#include <gamma.h>
#include <loop_coarse_ulocal.h>

template void createCoarseLoop_uLocal<double>(complex<double> *loop_h,
					      MG_Mugiq *mg_env, Eigsolve_Mugiq *eigsolve,
					      QudaInvertParam *invParams, MugiqLoopParam *loopParams);
template void createCoarseLoop_uLocal<float>(complex<float> *loop_h,
					     MG_Mugiq *mg_env, Eigsolve_Mugiq *eigsolve,
					     QudaInvertParam *invParams, MugiqLoopParam *loopParams);

template <typename Float>
void createCoarseLoop_uLocal(complex<Float> *loop_h,
			     MG_Mugiq *mg_env, Eigsolve_Mugiq *eigsolve,
			     QudaInvertParam *invParams, MugiqLoopParam *loopParams){

  //- Allocate the device loop buffer
  //- It will be overwritten for each momentum and time, hence it has only gamma-matrix dependence
  complex<Float> *loop_dev;
  size_t loopSize_dev = sizeof(complex<Float>) * N_GAMMA_;
  cudaMalloc((void**)&loop_dev, loopSize_dev);
  checkCudaError();
  
  QudaPrecision ePrec = eigsolve->getEvecs()[0]->Precision(); 
  if((ePrec != QUDA_DOUBLE_PRECISION) && (ePrec != QUDA_SINGLE_PRECISION))
    errorQuda("%s: Unsupported precision for creating Coarse part of loop\n", __func__);
  else printfQuda("%s: Working in %s precision\n", __func__, ePrec == QUDA_DOUBLE_PRECISION ? "double" : "single");
 
  //- Create the coefficients of the gamma matrices and copy them to __constant__ memory
  if((invParams->gamma_basis != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) &&
     (mg_env->mgParams->invert_param->gamma_basis != QUDA_DEGRAND_ROSSI_GAMMA_BASIS))
    errorQuda("%s: Supports only DeGrand-Rossi gamma basis\n", __func__);
  complex<Float> gCoeff[N_GAMMA_][SPINOR_SITE_LEN_*SPINOR_SITE_LEN_];
  createGammaCoeff<Float>(gCoeff);
  //-----------------------------------------------------------
  
  //- Allocate coarse gamma-matrix unit vectors
  int nUnit = SPINOR_SITE_LEN_;
  std::vector<ColorSpinorField*> unitGammaPos; // These are coarse fields (no phase)
  std::vector<ColorSpinorField*> unitGammaMom; // These are coarse fields with FT phase information
  
  
  ColorSpinorParam ucsParam(*(eigsolve->getEvecs()[0]));
  ucsParam.create = QUDA_ZERO_FIELD_CREATE;
  ucsParam.location = QUDA_CUDA_FIELD_LOCATION;
  ucsParam.setPrecision(ePrec);
  
  for(int n=0;n<nUnit;n++){
    unitGammaPos.push_back(ColorSpinorField::Create(ucsParam));
    unitGammaMom.push_back(ColorSpinorField::Create(ucsParam));
  }
  //-----------------------------------------------------------

  //- Create one temporary fine and N_coarse-1 temporary coarse fields
  //- Will be used for restricting the fine position space and phased
  //- Gamma generators 
  int nextCoarse = mg_env->nCoarseLevels - 1;
  
  std::vector<ColorSpinorField *> tmpCSF;
  ColorSpinorParam csParam(*(mg_env->mg_solver->B[0]));
  QudaPrecision coarsePrec = unitGammaPos[0]->Precision();
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  csParam.setPrecision(coarsePrec);
  
  tmpCSF.push_back(ColorSpinorField::Create(csParam)); //- tmpCSF[0] is a fine field
  for(int lev=0;lev<nextCoarse;lev++){
    tmpCSF.push_back(tmpCSF[lev]->CreateCoarse(mg_env->mgParams->geo_block_size[lev],
					       mg_env->mgParams->spin_block_size[lev],
					       mg_env->mgParams->n_vec[lev],
					       coarsePrec,
					       mg_env->mgParams->setup_location[lev+1]));
  }//-lev
  //-----------------------------------------------------------
  
  //- These are fine fields, will be used for both
  //- position space and phased generators
  std::vector<ColorSpinorField*> gammaGens;
  std::vector<ColorSpinorField*> gammaGensTD;
  gammaGens.resize(nUnit);
  gammaGensTD.resize(nUnit);

  const int *localL = mg_env->mg_solver->B[0]->X(); //-local space-time coordinates
  
  ColorSpinorParam cpuParam(NULL, *invParams, localL,
			    invParams->solution_type, invParams->input_location);
  cpuParam.fieldOrder = unitGammaPos[0]->FieldOrder();
  cpuParam.siteOrder  = unitGammaPos[0]->SiteOrder();
  cpuParam.setPrecision(unitGammaPos[0]->Precision());
  ColorSpinorParam cudaParam(cpuParam);
  cudaParam.fieldOrder = unitGammaPos[0]->FieldOrder();
  cudaParam.siteOrder  = unitGammaPos[0]->SiteOrder();
  cudaParam.location   = QUDA_CUDA_FIELD_LOCATION;
  cudaParam.create     = QUDA_ZERO_FIELD_CREATE;
  cudaParam.setPrecision(unitGammaPos[0]->Precision());
  for(int n=0;n<nUnit;n++){
    gammaGens[n]   = ColorSpinorField::Create(cudaParam);
    gammaGensTD[n] = ColorSpinorField::Create(cudaParam);
  }
  //-----------------------------------------------------------

  //- Create the unphased Gamma unity vectors
  createUnphasedGammaUnitVectors<Float>(gammaGens);
  printfQuda("%s: Unphased Coarse Gamma Vectors created\n", __func__);
  
  //- Restrict unphased fine gamma Generators at the coarsest level
  restrictGammaUnitVectors(unitGammaPos, gammaGens, tmpCSF, mg_env);
  printfQuda("%s: Restricting Unphased Gamma Vectors at the coarsest level completed\n", __func__);

  
  for(int p=0;p<loopParams->Nmom;p++){
    std::vector<int> mom = loopParams->momMatrix[p];
    printfQuda("%s: Performing Coarse part of the loop for momentum (%+02d,%+02d,%+02d)\n", __func__, mom[0], mom[1], mom[2]);
    
    //- Create the phased Gamma unity vectors
    createPhasedGammaUnitVectors<Float>(gammaGens, mom, loopParams->FTSign);
    printfQuda("%s: Phased Coarse Gamma Vectors created\n", __func__);
    
    /* The restrictor on the phased Gamma generators must run only on the 3d volume (no time). However,
     * the QUDA restrictor runs on full volume. Therefore, we perform here a dilution over the time direction,
     * so that the QUDA restrictor is called Lt times, where Lt is the global time size. In this way, the QUDA
     * restrictor only has effect on the single non-zero time slice.
     */
    const int globT = localL[3] * comm_dim(3);
    for(int gt=0;gt<globT;gt++){
      //-Perform time-dilution of gamma generators
      timeDilutePhasedGammaUnitVectors<Float>(gammaGensTD, gammaGens, gt);
      
      //- Restrict phased and time-diluted fine gamma Generators at the coarsest level
      restrictGammaUnitVectors(unitGammaMom, gammaGensTD, tmpCSF, mg_env);
      printfQuda("%s: Diluting and restricting Phased Coarse Gamma Vectors at the coarsest level for t = %d completed\n", __func__, gt);
      
      //- Call top-level function that calls CUDA kernel to assemble final loop
      cudaMemset(loop_dev,0, loopSize_dev);
      assembleCoarseLoop_uLocal<Float>(loop_dev, mg_env, eigsolve, unitGammaPos, unitGammaMom, invParams, loopParams);

      //- Copy device buffer back to appropriate place in host buffer
      //- Host loop buffer runs gamma-inside-time-inside-momentum g + Ng*t + Ng*Nt*p
      int hIdx = N_GAMMA_*gt + N_GAMMA_*globT*p;
      cudaMemcpy(&(loop_h[hIdx]), loop_dev, loopSize_dev, cudaMemcpyDeviceToHost);
      checkCudaError();
      
    }//- time
    
    printfQuda("%s: Coarse part of the loop done for momentum (%+02d,%+02d,%+02d)\n", __func__, mom[0], mom[1], mom[2]);
  }//- momentum

    
  //- Clean-up
  int nTmp = static_cast<int>(tmpCSF.size());
  for(int i=0;i<nTmp;i++)  delete tmpCSF[i];
  for(int n=0;n<nUnit;n++){
    delete gammaGens[n];
    delete gammaGensTD[n];
    delete unitGammaPos[n];
    delete unitGammaMom[n];
  }

  cudaFree(loop_dev);
  loop_dev = nullptr;
  
}
//-------------------------------------------------------------------------------
