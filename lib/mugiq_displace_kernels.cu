#include <mugiq_displace_kernels.cuh>
/*
//- Whether a site is even (return 0) or odd (return 1)
inline static __device__ int everORodd(const int x[]){
  return (x[0] + x[1] + x[2] + x[3]) % 2;
}

template <typename Float>
inline static __device__ Link<Float> getNbrLink(Gauge<Float> &U, const int coord[], int pty,
						int dir, DisplaceSign dispSign,
						const int dim[], const int commDim[], const int nFace){

  int nbrPty = 1 - pty; //- Parity of neighboring site
  
  Link<Float> dispU;
  if (dispSign == DispSignPlus){ //- dispU_d(x) <- U_d(x) (no need to take neighbouring site)
    dispU = U(dir, linkIndex(coord, dim), pty);
  }
  else if(dispSign == DispSignMinus){ //- dispU_d(x) <- U_d^\dag(x-d)
    //- We are at the left boundary, get backward neighbouring site from the halos
    if (commDim[dir] && (coord[dir] - nFace < 0)) {
      const int ghostIdx = ghostFaceIndex<0>(coord, dim, dir, nFace);
      const Link<Float> U2 = U.Ghost(dir, ghostIdx, nbrPty);
      dispU = conj(U2);
    }
    //- Not at the boundary
    else{
      const int bwdIdx = linkIndexM1(coord, dim, dir);
      const Link<Float> U2 = U(dir, bwdIdx, nbrPty);
      dispU = conj(U2);
    }
  }
  return dispU;
}
//-------------------------------------------------------------------
//-------------------------------------------------------------------


template <typename Float>
inline static __device__ Link<Float> getNbrLinkDispExtG(Gauge<Float> &U, const int coord[], int pty,
							const int dx[],
							int dir, DisplaceSign dispSign,
							const int dimEx[], const int brd[]){
  int dx1[5] = {0,0,0,0,0};
  int c2[5]  = {0,0,0,0,0};
  for (int i=0;i<N_DIM_;i++){
    dx1[i] = dx[i];
    c2[i]  = coord[i] + brd[i];
  }

  //- For positive displacements we stay on the current site
  //- For negative displacements, move one site backwards in the "dir" direction  
  if(dispSign == DispSignMinus) dx1[dir] -= 1;
  
  int nbrPty = (everORodd(dx1) == 0) ? pty : 1 - pty; //- Parity of neighboring site (even or odd)
  
  Link<Float> dispU;
  if (dispSign == DispSignPlus)
    dispU = U(dir, linkIndexShift(c2, dx1, dimEx), nbrPty); //- dispU_d(x) <- U_d(x)
  else{
    const Link<Float> U2 = U(dir, linkIndexShift(c2, dx1, dimEx), nbrPty);  //- U2_d(x) <- U_d(x-d)
    dispU = conj(U2);                                                       //- dispU_d(x) <- U_d^\dag(x-d)
  }
  
  return dispU;
}

template <typename Float>
inline static __device__ Link<Float> getNbrLinkExtG(Gauge<Float> &U, const int coord[], int pty,
						    int dir, DisplaceSign dispSign,
						    const int dimEx[], const int brd[]){
  int dx[5] = {0,0,0,0,0};
  return getNbrLinkDispExtG<Float>(U, coord, pty, dx, dir, dispSign, dimEx, brd);
}
//-------------------------------------------------------------------
//-------------------------------------------------------------------


template <typename Float>
inline static __device__ Vector<Float> getNbrSiteVec(Fermion<Float> &F, const int coord[], int pty,
						     int dir, DisplaceSign dispSign,
						     const int dim[], const int commDim[], const int nFace){

  int nbrPty = 1 - pty; //- Parity of neighboring site
  
  Vector<Float> dispV;
  if (dispSign == DispSignPlus){ //- dispV <- F(x+d)
    //- We are at the right boundary, get forward neighbouring site from the halos
    if (commDim[dir] && (coord[dir] + nFace >= dim[dir]) ) { 
      const int ghostIdx = ghostFaceIndex<1>(coord, dim, dir, nFace);
      dispV = F.Ghost(dir, 1, ghostIdx, nbrPty);
    }
    //- Not at the boundary
    else{ 
      const int fwdIdx = linkIndexP1(coord, dim, dir);
      dispV = F(fwdIdx, nbrPty);
    }
  }
  else if(dispSign == DispSignMinus){ //- dispV <- F(x-d)
    //- We are at the left boundary, get backward neighbouring site from the halos
    if (commDim[dir] && (coord[dir] - nFace < 0)) {  
      const int ghostIdx = ghostFaceIndex<0>(coord, dim, dir, nFace);
      dispV = F.Ghost(dir, 0, ghostIdx, nbrPty);
    }
    //- Not at the boundary
    else{
      const int bwdIdx = linkIndexM1(coord, dim, dir);
      dispV = F(bwdIdx, nbrPty);
    }
  }
  return dispV;
}
//-------------------------------------------------------------------
//-------------------------------------------------------------------

*/
template <typename Float, typename Arg>
__global__ void covariantDisplacementVector_kernel(Arg *arg,
						   DisplaceDir dispDir, DisplaceSign dispSign){
  /*
  int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
  int pty  = blockIdx.y*blockDim.y + threadIdx.y;
  pty = (arg->nParity == 2) ? pty : arg->parity;
  if (x_cb >= arg->volumeCB) return;
  if (pty >= arg->nParity) return;

  //- Get the local coordinates (must be 5d in case of DW fermions, not applicable here)
  int coord[5];
  getCoords(coord, x_cb, arg->dim, pty);
  coord[4] = 0;

  int dir = (int)dispDir; //- Direction of the displacement (0:x, 1:y, 2:z, 3:t)

  //- The neighbouring vector of site x, V(x+d) or V(x-d)
  Vector<Float> nbrV = getNbrSiteVec<Float>(arg->src, coord, pty, dir, dispSign, arg->dim, arg->commDim, arg->nFace);

  Link<Float> nbrU; //- Neighbouring Link, U_d(x) or U_d^\dag(x-d)
  if(arg->extendedGauge)
    nbrU = getNbrLinkExtG<Float>(arg->U, coord, pty, dir, dispSign, arg->dimEx, arg->brd);
  else
    nbrU = getNbrLink<Float>(arg->U, coord, pty, dir, dispSign, arg->dim, arg->commDim, arg->nFace);

  arg->dst(x_cb, pty) = nbrU * nbrV; // dst(x) = U_d(x) * V(x+d) || U_d^\dag(x-d) * V(x-d)
  */
}

template __global__ void covariantDisplacementVector_kernel<float, CovDispVecArg<float,QUDA_FLOAT2_FIELD_ORDER>>
(CovDispVecArg<float, QUDA_FLOAT2_FIELD_ORDER> *arg, DisplaceDir dispDir, DisplaceSign dispSign);
template __global__ void covariantDisplacementVector_kernel<float, CovDispVecArg<float,QUDA_FLOAT4_FIELD_ORDER>>
(CovDispVecArg<float, QUDA_FLOAT4_FIELD_ORDER> *arg, DisplaceDir dispDir, DisplaceSign dispSign);
template __global__ void covariantDisplacementVector_kernel<double, CovDispVecArg<double,QUDA_FLOAT2_FIELD_ORDER>>
(CovDispVecArg<double, QUDA_FLOAT2_FIELD_ORDER> *arg, DisplaceDir dispDir, DisplaceSign dispSign);
template __global__ void covariantDisplacementVector_kernel<double, CovDispVecArg<double,QUDA_FLOAT4_FIELD_ORDER>>
(CovDispVecArg<double, QUDA_FLOAT4_FIELD_ORDER> *arg, DisplaceDir dispDir, DisplaceSign dispSign);
