#include <disp_state.h>



template <typename T>
LoopDispState<T>::LoopDispState(MugiqLoopParam *loopParams_) :
  gaugePtr{loopParams_->gauge[0],loopParams_->gauge[1],loopParams_->gauge[2],loopParams_->gauge[3]}
{
  
}


template <typename T>
LoopDispState<T>::~LoopDispState()
{

  for(int i=0;i<N_DIM_;i++) gaugePtr[i] = nullptr;
  
}









template class LoopDispState<float>;
template class LoopDispState<double>;
