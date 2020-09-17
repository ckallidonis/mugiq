#include <displace.h>


template <typename T>
Displace<T>::Displace(MugiqLoopParam *loopParams_) :
  dispStr('\0'), prevDispStr('\0'),
  dispFlag(DispFlag_None), dispDir(DispDirNone), dispSign(DispSignNone),
  gaugePtr{loopParams_->gauge[0],loopParams_->gauge[1],loopParams_->gauge[2],loopParams_->gauge[3]},
  qGaugePrm(loopParams_->gauge_param),
  gaugeField(nullptr),
  dispVec(nullptr)
{

  for (int d=0;d<N_DIM_;d++) exRng[d] = 2 * (redundantComms || commDimPartitioned(d));

  //-Create the gauge field with extended ghost exchange, will be used for displacements
  createExtendedCudaGaugeField();
  
}


template <typename T>
Displace<T>::~Displace()
{

  for(int i=0;i<N_DIM_;i++) gaugePtr[i] = nullptr;

  if(gaugeField) delete gaugeField;
}


template <typename T>
cudaGaugeField* Displace<T>::createCudaGaugeField(){

  cudaGaugeField *cudaGaugeField = NULL;
  
  GaugeFieldParam gParam(gaugePtr, *qGaugePrm);
  GaugeField *cpuGaugeField = static_cast<GaugeField*>(new quda::cpuGaugeField(gParam));

  gParam.create         = QUDA_NULL_FIELD_CREATE;
  gParam.reconstruct    = qGaugePrm->reconstruct;
  gParam.ghostExchange  = QUDA_GHOST_EXCHANGE_PAD;
  gParam.pad            = qGaugePrm->ga_pad * 2;  // It's originally defined with half-volume
  gParam.order          = QUDA_QDP_GAUGE_ORDER;//QUDA_FLOAT2_GAUGE_ORDER;

  if((qGaugePrm->cuda_prec == QUDA_SINGLE_PRECISION && typeid(T) != typeid(float)) ||
     (qGaugePrm->cuda_prec == QUDA_DOUBLE_PRECISION && typeid(T) != typeid(double)))
    errorQuda("%s: Incompatible precision settings between Displace template and gauge field parameters\n");
  
  gParam.setPrecision(qGaugePrm->cuda_prec, true);  
  
  cudaGaugeField = new quda::cudaGaugeField(gParam);
  if(cudaGaugeField == NULL) return NULL;

  if(gaugePtr != NULL) cudaGaugeField->copy(*cpuGaugeField); // C.K. This does ghost exchange as well

  delete cpuGaugeField;
  cpuGaugeField = NULL;

  return cudaGaugeField;  
}



template <typename T>
void Displace<T>::createExtendedCudaGaugeField(bool copyGauge, bool redundant_comms, QudaReconstructType recon){

  cudaGaugeField *tmpGauge = createCudaGaugeField();
 
  int y[4];
  for (int dir=0;dir<N_DIM_;dir++) y[dir] = tmpGauge->X()[dir] + 2*exRng[dir];
  int pad = 0;

  GaugeFieldParam gParamEx(y, tmpGauge->Precision(), recon != QUDA_RECONSTRUCT_INVALID ? recon : tmpGauge->Reconstruct(), pad,
			   tmpGauge->Geometry(), QUDA_GHOST_EXCHANGE_EXTENDED);
  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
  gParamEx.order = tmpGauge->Order();
  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParamEx.t_boundary = tmpGauge->TBoundary();
  gParamEx.nFace = 1;
  gParamEx.tadpole = tmpGauge->Tadpole();
  for (int d=0; d<N_DIM_; d++) gParamEx.r[d] = exRng[d];

  gaugeField = new cudaGaugeField(gParamEx);

  if(copyGauge){
    copyExtendedGauge(*gaugeField, *tmpGauge, QUDA_CUDA_FIELD_LOCATION);
    gaugeField->exchangeExtendedGhost(exRng, redundant_comms);
  }

  delete tmpGauge;
  tmpGauge = nullptr;

  printfQuda("%s: Extended Gauge Field for Displacements created\n", __func__);
}


template <typename T>
DisplaceFlag Displace<T>::ParseDisplaceFlag(){

  DisplaceFlag dFlag = DispFlag_None;
  for(int iopt=0;iopt<N_DISPLACE_FLAGS;iopt++){
    if( dispStr == DisplaceFlagArray[iopt] ){
      dFlag = (DisplaceFlag)iopt;
      break;
    }
  }
  if(dFlag == DispFlag_None) errorQuda("%s: Cannot parse given displacement string = %c.\n", __func__, dispStr);
  return dFlag;
}


template <typename T>
DisplaceDir Displace<T>::ParseDisplaceDir(){

  DisplaceDir dDir = DispDirNone;
  switch(dispFlag){
  case DispFlag_x:
  case DispFlag_X: {
    dDir = DispDir_x;
  } break;
  case DispFlag_y:
  case DispFlag_Y: {
    dDir = DispDir_y;
  } break;
  case DispFlag_z:
  case DispFlag_Z: {
    dDir = DispDir_z;
  } break;
  case DispFlag_t:
  case DispFlag_T: {
    dDir = DispDir_t;
  } break;
  default: errorQuda("%s: Unsupported displacement flag, dispFlag = %c.\n",
		     __func__, (dispFlag >=0 && dispFlag<N_DISPLACE_FLAGS) ? DisplaceFlagArray[(int)dispFlag] : '?');
    }//-- switch

  return dDir;
  
}

template <typename T>
DisplaceSign Displace<T>::ParseDisplaceSign(){
  
  DisplaceSign dSign = DispSignNone;
  switch(dispFlag){
  case DispFlag_X:
  case DispFlag_Y:
  case DispFlag_Z:
  case DispFlag_T: {
    dSign = DispSignPlus;
  } break;
  case DispFlag_x:
  case DispFlag_y:
  case DispFlag_z:
  case DispFlag_t: {
    dSign = DispSignMinus;
  } break;
  default: errorQuda("%s: Unsupported displace flag, dispFlag = %c.\n",
		     __func__, (dispFlag >=0 && dispFlag<N_DISPLACE_FLAGS) ? DisplaceFlagArray[(int)dispFlag] : '?');
  }//-- switch
  
  return dSign;
}



template <typename T>
void Displace<T>::setupDisplacement(char dStr){

  dispStr = dStr;
  
  dispFlag = ParseDisplaceFlag();
  dispDir  = ParseDisplaceDir();
  dispSign = ParseDisplaceSign();
  
  if( ((int)dispSign>=0 && (int)dispSign<N_DISPLACE_SIGNS) && ((int)dispDir>=0 && (int)dispDir<N_DIM_)  ){
    if(getVerbosity() >= QUDA_VERBOSE)
      printfQuda("%s: Displacement is in the %s%s direction\n",
		 __func__, DisplaceSignArray[(int)dispSign], DisplaceDirArray[(int)dispDir]);
  }
  else
    errorQuda("%s: Got invalid dispDir and/or dispSign.\n", __func__);

}






template class Displace<float>;
template class Displace<double>;
