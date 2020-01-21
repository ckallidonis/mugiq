#ifndef _GAMMA_MUGIQ_H
#define _GAMMA_MUGIQ_H

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

#define N_SPIN_ 4
#define N_COLOR_ 3
#define N_GAMMA_ 16
#define SPINOR_SITE_LEN_ (N_SPIN_ * N_COLOR_)
#define GAUGE_SITE_LEN_ (N_COLOR_ * N_COLOR_)
#define GAMMA_LEN_ (N_SPIN_ * N_SPIN_)

#define MUGIQ_MAX_FINE_VEC 24
#define MUGIQ_MAX_COARSE_VEC 256

#define THREADS_PER_BLOCK 64

using namespace quda;


template <QudaPrecision prec> struct FieldMapper {};

template <> struct FieldMapper<QUDA_DOUBLE_PRECISION> {
  typedef typename colorspinor_mapper<double, N_SPIN_, N_COLOR_>::type FermionField;
  typedef ColorSpinor<double, N_SPIN_, N_COLOR_> Vector;
  
  typedef typename gauge_mapper<double, QUDA_RECONSTRUCT_NO>::type GaugField;
  typedef Matrix<complex<double>, N_COLOR_> Link;
};


typedef typename colorspinor_mapper<double,N_SPIN_,N_COLOR_>::type Fermion;

#if 0
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

#if 1

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
template <QudaPrecision prec>
struct Arg_Gamma : public ArgGeom {
  
  //  typename FieldMapper<prec>::FermionField gammaGens[MUGIQ_MAX_FINE_VEC];
  typename FieldMapper<prec>::FermionField testSpinor;
  //Fermion testSpinor;
  
  int nVec;
  
  Arg_Gamma () {}
  
  Arg_Gamma(std::vector<ColorSpinorField*> &gammaGens_)
    : ArgGeom(gammaGens_[0]),
      testSpinor(*gammaGens_[0]),
      nVec(gammaGens_.size())
  {
  }
  
};

#endif


#endif // _GAMMA_MUGIQ_H
