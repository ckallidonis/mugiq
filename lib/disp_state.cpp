#include <disp_state.h>


template <typename T>
LoopDispState<T>::LoopDispState(MugiqLoopParam *loopParams_) :
  gaugePtr{loopParams_->gauge[0],loopParams_->gauge[1],loopParams_->gauge[2],loopParams_->gauge[3]},
  qGaugePrm(loopParams_->gauge_param)
{

  for (int d=0;d<N_DIM_;d++) exRng[d] = 2 * (redundantComms || commDimPartitioned(d));

  //-Create the gauge field with extended ghost exchange, will be used for displacements
  createExtendedCudaGaugeField();
}


template <typename T>
LoopDispState<T>::~LoopDispState()
{

  for(int i=0;i<N_DIM_;i++) gaugePtr[i] = nullptr;

  if(gaugeField) delete gaugeField;
}


template <typename T>
cudaGaugeField* LoopDispState<T>::createCudaGaugeField(){

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
    errorQuda("%s: Incompatible precision settings between LoopDispState template and gauge field parameters\n");
  
  gParam.setPrecision(qGaugePrm->cuda_prec, true);  
  
  cudaGaugeField = new quda::cudaGaugeField(gParam);
  if(cudaGaugeField == NULL) return NULL;

  if(gaugePtr != NULL) cudaGaugeField->copy(*cpuGaugeField); // C.K. This does ghost exchange as well

  delete cpuGaugeField;
  cpuGaugeField = NULL;

  return cudaGaugeField;  
}



template <typename T>
void LoopDispState<T>::createExtendedCudaGaugeField(bool copyGauge, bool redundant_comms, QudaReconstructType recon){

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






template class LoopDispState<float>;
template class LoopDispState<double>;
