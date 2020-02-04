#include <utility_kernels.h>

//- Create Gamma matrix generators in position space
template __global__ void createGammaGeneratorsPos_kernel<double>(ArgGammaPos<double> *arg);
template __global__ void createGammaGeneratorsPos_kernel<float>(ArgGammaPos<float> *arg);

template <typename Float>
__global__ void createGammaGeneratorsPos_kernel(ArgGammaPos<Float> *arg){

  int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
  int pty  = blockIdx.y*blockDim.y + threadIdx.y;
  pty = (arg->nParity == 2) ? pty : arg->parity;
  if (x_cb >= arg->volumeCB) return;
  if (pty >= arg->nParity) return;

  complex<Float> c1 = complex<Float>{1.0,0.0};

#pragma unroll
  for(int is=0;is<N_SPIN_;is++){     //- Which of the
#pragma unroll
    for(int ic=0;ic<N_COLOR_;ic++){  //- generator vectors (color-inside-spin)
      typename FieldMapper<Float>::Vector v;
#pragma unroll
      for(int js=0;js<N_SPIN_;js++){       //- Spin-color index
#pragma unroll
	for(int jc=0;jc<N_COLOR_;jc++){    //- within the vector (color-inside-spin)	  
	  if((is==js) && (ic==jc)) v(js,jc) = c1;
	  else v(js,jc) = 0.0;
	}//- jc
      }//- js
      int vIdx = GAMMA_GEN_IDX(is,ic);
      arg->gammaGensPos[vIdx](x_cb,pty) = v;
    }//- ic
  }//- is  
}


//- Create Gamma matrix generators with Exponential phase
template __global__ void createGammaGeneratorsMom_kernel<double>(ArgGammaMom<double> *arg);
template __global__ void createGammaGeneratorsMom_kernel<float>(ArgGammaMom<float> *arg);

template <typename Float>
__global__ void createGammaGeneratorsMom_kernel(ArgGammaMom<Float> *arg){

  int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
  int pty  = blockIdx.y*blockDim.y + threadIdx.y;
  pty = (arg->nParity == 2) ? pty : arg->parity;
  if (x_cb >= arg->volumeCB) return;
  if (pty >= arg->nParity) return;

  //- Local coordinates
  int lcoord[5];
  getCoords(lcoord, x_cb, arg->dim, pty);
  lcoord[4] = 0;

  //- Global 3-d coordinates (no time)
  int gcoord[3];
#pragma unroll
  for(int id=0;id<arg->d_mom;id++)
    gcoord[id] = lcoord[id] + arg->localL[id] * arg->commCoord[id];

  //- Construct phase
  Float expon = 0.0;
#pragma unroll
  for(int id=0;id<arg->d_mom;id++)
    expon += arg->mom[id]*gcoord[id] / static_cast<Float>(arg->globalL[id]);  

  complex<Float> phase = {cos(2.0*PI*expon), arg->FTSign*sin(2.0*PI*expon)};
  
#pragma unroll
  for(int is=0;is<N_SPIN_;is++){     //- Which of the
#pragma unroll
    for(int ic=0;ic<N_COLOR_;ic++){  //- generator vectors (color-inside-spin)
      typename FieldMapper<Float>::Vector v;
#pragma unroll
      for(int js=0;js<N_SPIN_;js++){       //- Spin-color index
#pragma unroll
	for(int jc=0;jc<N_COLOR_;jc++){    //- within the vector (color-inside-spin)	  
	  if((is==js) && (ic==jc)) v(js,jc) = phase;
	  else v(js,jc) = 0.0;
	}//- jc
      }//- js
      int vIdx = GAMMA_GEN_IDX(is,ic);
      arg->gammaGensMom[vIdx](x_cb,pty) = v;
    }//- ic
  }//- is  
}
