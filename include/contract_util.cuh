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

constexpr int cSize = 4096; // Size of constant memory symbols

__constant__ char cGamma[cSize]; // constant buffer for gamma matrices on GPU


//- Templates of the Fermion/Gauge mappers on the precision, used for fine fields
template <typename T> struct FieldMapper {};

template <> struct FieldMapper<double> {
  typedef typename colorspinor_mapper<double, N_SPIN_, N_COLOR_>::type FermionField;
  typedef ColorSpinor<double, N_SPIN_, N_COLOR_> Vector;

  typedef typename gauge_mapper<double, QUDA_RECONSTRUCT_NO>::type GaugeField;
  typedef Matrix<complex<double>, N_COLOR_> Link;
};

template <> struct FieldMapper<float> {
  typedef typename colorspinor_mapper<float, N_SPIN_, N_COLOR_>::type FermionField;
  typedef ColorSpinor<float, N_SPIN_, N_COLOR_> Vector;

  typedef typename gauge_mapper<float, QUDA_RECONSTRUCT_NO>::type GaugeField;
  typedef Matrix<complex<float>, N_COLOR_> Link;
};




/**
 * Hard-coded gamma coefficients for the DeGrand-Rossi basis
 * Gamma-index notation is: G{x,y,z,t} = G{1,2,3,4}
 * Gamma matrices are defined as: G(n) = g1^n0 . g2^n1 . g3^n2 . g4^n3, where n = n0*2^0 + n1*2^1 + n2*2^2 + n3*2^3
 * This parametrization helps in efficient unrolling and usage when performing trace contractions on the GPU,
 * taking into account only non-zero elements when performing the relevant summations of the Traces.
 * Any Gamma-matrix element can be obtained as: G(n)_{ij} = RowValue[n][i] * (ColumnIndex[n][i]==j ? 1 : 0)
 */

//- The value in rows 0,1,2,3, respectively, of each gamma matrix
constexpr int GammaRowValue[N_GAMMA_][N_SPIN_][2] = {{ {1,0}, {1,0}, {1,0}, {1,0} },   /* G0 = 1 */
						     { {0,1}, {0,1},{0,-1},{0,-1} },   /* G1 = g1 */
						     {{-1,0}, {1,0}, {1,0},{-1,0} },   /* G2 = g2 */
						     {{0,-1}, {0,1},{0,-1}, {0,1} },   /* G3 = g1 g2 */
						     { {0,1},{0,-1},{0,-1}, {0,1} },   /* G4 = g3 */
						     {{-1,0}, {1,0},{-1,0}, {1,0} },   /* G5 = g1 g3 */
						     {{0,-1},{0,-1},{0,-1},{0,-1} },   /* G6 = g2 g3 */
						     { {1,0}, {1,0},{-1,0},{-1,0} },   /* G7 = g1 g2 g3 */
						     { {1,0}, {1,0}, {1,0}, {1,0} },   /* G8 = g4 */
						     { {0,1}, {0,1},{0,-1},{0,-1} },   /* G9 = g1 g4 */
						     {{-1,0}, {1,0}, {1,0},{-1,0} },   /* G10= g2 g4 */
						     {{0,-1}, {0,1},{0,-1}, {0,1} },   /* G11= g1 g2 g4 */
						     { {0,1},{0,-1},{0,-1}, {0,1} },   /* G12= g3 g4 */
						     {{-1,0}, {1,0},{-1,0}, {1,0} },   /* G13= g1 g3 g4 */
						     {{0,-1},{0,-1},{0,-1},{0,-1} },   /* G14= g2 g3 g4 */
						     { {1,0}, {1,0},{-1,0},{-1,0} }};  /* G15= g1 g2 g3 g4 */


//- The column in which RowValue exists for each gamma matrix
constexpr int GammaColumnIndex[N_GAMMA_][N_SPIN_] = {{ 0, 1, 2, 3 },   /* G0 = 1 */
						     { 3, 2, 1, 0 },   /* G1 = g1 */
						     { 3, 2, 1, 0 },   /* G2 = g2 */
						     { 0, 1, 2, 3 },   /* G3 = g1 g2 */
						     { 2, 3, 0, 1 },   /* G4 = g3 */
						     { 1, 0, 3, 2 },   /* G5 = g1 g3 */
						     { 1, 0, 3, 2 },   /* G6 = g2 g3 */
						     { 2, 3, 0, 1 },   /* G7 = g1 g2 g3 */
						     { 2, 3, 0, 1 },   /* G8 = g4 */
						     { 1, 0, 3, 2 },   /* G9 = g1 g4 */
						     { 1, 0, 3, 2 },   /* G10= g2 g4 */
						     { 2, 3, 0, 1 },   /* G11= g1 g2 g4 */
						     { 0, 1, 2, 3 },   /* G12= g3 g4 */
						     { 3, 2, 1, 0 },   /* G13= g1 g3 g4 */
						     { 3, 2, 1, 0 },   /* G14= g2 g3 g4 */
						     { 0, 1, 2, 3 }};  /* G15= g1 g2 g3 g4 */


//- Structure that will eventually be copied to GPU __constant__ memory
template <typename Float>
struct GammaCoeff{
  complex<Float> row_value[N_GAMMA_][N_SPIN_];
  int column_index[N_GAMMA_][N_SPIN_];
}; // GammaCoeff



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
  
  ArgGeom(ColorSpinorField *x)
    : parity(0), nParity(x->SiteSubset()), nFace(1),
      dim{ (3-nParity) * x->X(0), x->X(1), x->X(2), x->X(3), 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      lL{x->X(0), x->X(1), x->X(2), x->X(3)},
      volumeCB(x->VolumeCB()), volume(x->Volume())
  { }
  
  ArgGeom(cudaGaugeField *u)
    : parity(0), nParity(u->SiteSubset()), nFace(1),
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      lL{u->X()[0], u->X()[1], u->X()[2], u->X()[3]}
  {
    if(u->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED){
      volume = 1;
      for(int dir=0;dir<4;dir++){
	dim[dir] = u->X()[dir] - 2*u->R()[dir];   //-- Actual lattice dimensions (NOT extended)
	dimEx[dir] = dim[dir] + 2*u->R()[dir];    //-- Extended lattice dimensions
	brd[dir] = u->R()[dir];
	volume *= dim[dir];
      }
      volumeCB = volume/2;
    }
    else{
      volume = 1;
      for(int dir=0;dir<4;dir++){
	dim[dir] = u->X()[dir];
	volume *= dim[dir];
      }
      volumeCB = volume/2;
      dim[0] *= (3-nParity);
    }
    dim[4] = 1;
  }
};//-- ArgGeom


//- Argument Structure for performing the loop contractions
template <typename Float>
struct LoopContractArg : public ArgGeom {

  typename FieldMapper<Float>::FermionField eVecL; //- Left  eigenvector in trace
  typename FieldMapper<Float>::FermionField eVecR; //- Right eigenvector in trace

  Float inv_sigma; //- The inverse(!) of the eigenvalue corresponding to eVecL and eVecR
  
  LoopContractArg(ColorSpinorField *eVecL_, ColorSpinorField *eVecR_, Float sigma)
    : ArgGeom(eVecL_), eVecL(*eVecL_), eVecR(*eVecR_), inv_sigma(1.0/sigma)
  { }
  
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



template <typename Float>
struct CovDispVecArg : public ArgGeom {

  typename FieldMapper<Float>::FermionField dst;
  typename FieldMapper<Float>::FermionField src;
  typename FieldMapper<Float>::GaugeField U;
  
  MuGiqBool extendedGauge;
  
  CovDispVecArg(ColorSpinorField *dst_, ColorSpinorField *src_, cudaGaugeField *U_)
    : ArgGeom(U_),
      dst(*dst_), src(*src_), U(*U_),
      extendedGauge((U_->GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED) ? MUGIQ_BOOL_TRUE : MUGIQ_BOOL_FALSE)
  { }

  ~CovDispVecArg() {}
};



#endif // _CONTRACT_UTIL_CUH
