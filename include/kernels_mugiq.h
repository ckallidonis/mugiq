#ifndef _KERNELS_MUGIQ_H
#define _KERNELS_MUGIQ_H

#include <mpi.h>
#include <transfer.h>
#include <complex_quda.h>
#include <quda_internal.h>
#include <quda_matrix.h>
#include <index_helper.cuh>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <util_quda.h>
#include <util_mugiq.h>

using namespace quda;


//- Templates of the Fermion/Gauge mappers on the precision
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


//- Base structure holding geometry-related parameters
struct ArgGeom {
  
  int parity;                 // hard code to 0 for now
  int nParity;                // number of parities we're working on
  int nFace;                  // hard code to 1 for now
  int dim[5];                 // full lattice dimensions
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


//- Structure used for creating the gamma matrix generators
template <typename T>
struct Arg_Gamma : public ArgGeom {

  typename FieldMapper<T>::FermionField gammaGens[SPINOR_SITE_LEN_];  
  int nVec;
  
  Arg_Gamma () {}

  Arg_Gamma(std::vector<ColorSpinorField*> &gammaGens_)
    : ArgGeom(gammaGens_[0]),
      nVec(gammaGens_.size())
  {
    if(nVec!=SPINOR_SITE_LEN_)
      errorQuda("%s: Size of Gamma generators must be Nspin*Ncolor = %d", __func__, SPINOR_SITE_LEN_);
    
    for(int ivec=0;ivec<nVec;ivec++)
      gammaGens[ivec].init(*gammaGens_[ivec]);
  }
  
};


//- Declaration of variables and functions related to gamma matrices
//- GPU Constant memory size for Gamma matrix coeffs
//- Size is Ngamma(16)*Nspin*Ncolor*Nspin*Ncolor*2(re-im)*8(bytes) = 36KB
constexpr int cSize_gamma = 36864;
__constant__ char gCoeff_cMem[cSize_gamma]; //- GPU Constant memory buffer for gamma coefficients


#endif // _KERNELS_MUGIQ_H



#if 0
  Arg_Gamma(ColorSpinorField **gammaGens_)
#endif

#if 0
//  std::vector<typename FieldMapper<T>::FermionField> gammaGens;

typedef typename colorspinor_mapper<double,N_SPIN_,N_COLOR_>::type Fermion;

typedef typename colorspinor_mapper<double,N_SPIN_,N_COLOR_>::type ColorSpinor_Double;
typedef typename colorspinor_mapper<float, N_SPIN_,N_COLOR_>::type ColorSpinor_Float;
typedef typename colorspinor_mapper<short, N_SPIN_,N_COLOR_>::type ColorSpinor_Half;
typedef ColorSpinor<double,N_SPIN_,N_COLOR_> Vector_Double;
typedef ColorSpinor<float, N_SPIN_,N_COLOR_> Vector_Single;
typedef ColorSpinor<short, N_SPIN_,N_COLOR_> Vector_Half;

typedef typename gauge_mapper<double,QUDA_RECONSTRUCT_NO>::type Gauge_Double;
typedef typename gauge_mapper<float,QUDA_RECONSTRUCT_NO>::type Gauge_Float;
typedef typename gauge_mapper<short,QUDA_RECONSTRUCT_NO>::type Gauge_Half;
typedef Matrix<complex<double>,N_COLOR_> Link_Double;
typedef Matrix<complex<float>, N_COLOR_> Link_Single;
typedef Matrix<complex<half>,  N_COLOR_> Link_Half;
#endif
