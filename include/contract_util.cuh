#ifndef _CONTRACT_UTIL_CUH
#define _CONTRACT_UTIL_CUH

#include <util_mugiq.h>
#include <mugiq.h>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>

using namespace quda;

constexpr int cSize = 8192; // Size of constant memory symbols, set it to 8K

__constant__ char cGammaCoeff[cSize];  //- constant-memory buffer for gamma matrices on GPU
__constant__ char cGammaMap[cSize];    //- constant-memory buffer for mapping Gamma to g5*Gamma


template <typename Float, QudaFieldOrder fieldOrder>
using Fermion = colorspinor::FieldOrderCB<Float, N_SPIN_, N_COLOR_, 1, fieldOrder>;

template <typename Float>
using Gauge = typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type;

template <typename Float>
using Vector = ColorSpinor<Float,N_COLOR_,N_SPIN_>;

template <typename Float>
using Link = Matrix<complex<Float>,N_COLOR_>;

//- Structure that will eventually be copied to GPU __constant__ memory
template <typename Float>
struct GammaCoeff{
  complex<Float> row_value[N_GAMMA_][N_SPIN_];
  int column_index[N_GAMMA_][N_SPIN_];
}; // GammaCoeff


//- Structure that will eventually be copied to GPU __constant__ memory
template <typename Float>
struct GammaMap{
  Float sign[N_GAMMA_];
  int index[N_GAMMA_];
}; // GammaMap



//- Argument structure for creating the Phase Matrix on GPU
struct MomProjArg{
  
  const int momDim = MOM_DIM_;
  const long long locV3;
  const int Nmom;
  const int FTSign;
  const int localL[N_DIM_];
  const int totalL[N_DIM_];
  const int commCoord[N_DIM_];
  
  MomProjArg(long long locV3_, int Nmom_, int FTSign_, const int localL_[], const int totalL_[])
    :   locV3(locV3_), Nmom(Nmom_), FTSign(FTSign_),
	localL{localL_[0],localL_[1],localL_[2],localL_[3]},
	totalL{totalL_[0],totalL_[1],totalL_[2],totalL_[3]},
	commCoord{comm_coord(0),comm_coord(1),comm_coord(2),comm_coord(3)}
  { }
}; //-- structure


//- Base argument structure holding geometry-related parameters
struct ArgGeom {
  
  int parity;                 // hard code to 0 for now
  int nParity;                // number of parities we're working on
  int nFace;                  // hard code to 1 for now
  int dim[5];                 // full (not checkerboard) local lattice dimensions
  int commDim[4];             // whether a given dimension is partitioned or not
  int lL[4];                  // 4-d local lattice dimensions
  int volumeCB;               // checkerboarded volume
  int volume;                 // full-site local volume
  
  int dimEx[4]; // extended Gauge field dimensions
  int brd[4];   // Border of extended gauge field (size of extended halos)
  
  ArgGeom () {}
  
  ArgGeom(ColorSpinorField &x)
    : parity(0), nParity(x.SiteSubset()), nFace(1),
      dim{ (3-nParity) * x.X(0), x.X(1), x.X(2), x.X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      lL{x.X(0), x.X(1), x.X(2), x.X(3)},
      volumeCB(x.VolumeCB()), volume(x.Volume())
  { }
  
  ArgGeom(cudaGaugeField &u)
    : parity(0), nParity(u.SiteSubset()), nFace(1),
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      lL{u.X()[0], u.X()[1], u.X()[2], u.X()[3]}
  {
    if(u.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED){
      volume = 1;
      for(int dir=0;dir<4;dir++){
	dim[dir] = u.X()[dir] - 2*u.R()[dir];   //-- Actual lattice dimensions (NOT extended)
	dimEx[dir] = dim[dir] + 2*u.R()[dir];    //-- Extended lattice dimensions
	brd[dir] = u.R()[dir];
	volume *= dim[dir];
      }
      volumeCB = volume/2;
    }
    else{
      volume = 1;
      for(int dir=0;dir<4;dir++){
	dim[dir] = u.X()[dir];
	volume *= dim[dir];
      }
      volumeCB = volume/2;
      dim[0] *= (3-nParity);
    }
    dim[4] = 1;
  }
};//-- ArgGeom


//- Argument Structure for performing the loop contractions
template <typename Float, QudaFieldOrder fieldOrder>
struct LoopContractArg : public ArgGeom {

  Fermion<Float, fieldOrder> eVecL; //- Left  eigenvector in trace
  Fermion<Float, fieldOrder> eVecR; //- Right eigenvector in trace

  Float inv_sigma; //- The inverse(!) of the eigenvalue corresponding to eVecL and eVecR
  
  LoopContractArg(ColorSpinorField &eVecL_, ColorSpinorField &eVecR_, Float sigma)
    : ArgGeom(eVecL_), eVecL(eVecL_), eVecR(eVecR_), inv_sigma(1.0/sigma)
  {  }
  
};//-- LoopContractArg


//- Structure used for index-converting kernel
struct ConvertIdxArg{
  
  const int tAxis = T_AXIS_;    // direction of the time-axis
  const int nData;              // Number of total data = nLoop * N_GAMMA
  const int nLoop;              // Number of loops in the input/output buffers
  const int nParity;            // number of parities we're working on
  const int volumeCB;           // checkerboarded volume
  const int localL[4];          // 4-d local lattice dimensions

  int stride_3d[4];             // stride in spatial volume
  int locV3;                    // spatial volume
  
  ConvertIdxArg(int nData_, int nLoop_, int nParity_, int volumeCB_, const int localL_[])
    : nData(nData_), nLoop(nLoop_), nParity(nParity_), volumeCB(volumeCB_),
      localL{localL_[0], localL_[1], localL_[2], localL_[3]},
      stride_3d{0,0,0,0}, locV3(0)
  {
    
    if(tAxis >= 0){
      int mul3d = 1;
      for(int i=0;i<N_DIM_;i++){
	if(i == tAxis) stride_3d[i] = 0;
	else{
	  stride_3d[i] = mul3d;
	  mul3d *= localL[i];
	}
	locV3 = mul3d;
      }//-for
    }//-if
    
  }//-- constructor
  
  ~ConvertIdxArg() {}
  
};//-- Structure definition



template <typename Float, QudaFieldOrder order>
struct CovDispVecArg : public ArgGeom {

  Fermion<Float, order> dst;
  Fermion<Float, order> src;
  Gauge<Float> U;
  
  MuGiqBool extendedGauge;
  
  CovDispVecArg(ColorSpinorField &dst_, ColorSpinorField &src_, cudaGaugeField &U_)
    : ArgGeom(U_),
      dst(dst_), src(src_), U(U_),
      extendedGauge((U_.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? MUGIQ_BOOL_TRUE : MUGIQ_BOOL_FALSE)
  { }

  ~CovDispVecArg() {}
};



#endif // _CONTRACT_UTIL_CUH
