#include <kernel_util.h>

template __global__ void createGammaGenerators_kernel<double>(Arg_Gamma<double> *arg);
template __global__ void createGammaGenerators_kernel<float>(Arg_Gamma<float> *arg);

template <typename T>
__global__ void createGammaGenerators_kernel(Arg_Gamma<T> *arg){

  int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
  int pty  = blockIdx.y*blockDim.y + threadIdx.y;
  pty = (arg->nParity == 2) ? pty : arg->parity;
  if (x_cb >= arg->volumeCB) return;
  if (pty >= arg->nParity) return;

  complex<T> c1 = complex<T>{1.0,0.0};

#pragma unroll
  for(int is=0;is<N_SPIN_;is++){     //- Which of the
#pragma unroll
    for(int ic=0;ic<N_COLOR_;ic++){  //- generator vectors (color-inside-spin)
      typename FieldMapper<T>::Vector v;
#pragma unroll
      for(int js=0;js<N_SPIN_;js++){       //- Spin-color index
#pragma unroll
	for(int jc=0;jc<N_COLOR_;jc++){    //- within the vector (color-inside-spin)	  
	  if((is==js) && (ic==jc)) v(js,jc) = c1;
	  else v(js,jc) = 0.0;
	}//- jc
      }//- js
      int vIdx = GAMMA_GEN_IDX(is,ic);
      arg->gammaGens[vIdx](x_cb,pty) = v;
    }//- ic
  }//- is  
}
