#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <complex.h>

#include <quda.h>
#include <quda_fortran.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <eigensolve_quda.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <llfat_quda.h>
#include <unitarization_links.h>
#include <algorithm>
#include <staggered_oprod.h>
#include <ks_improved_force.h>
#include <ks_force_quda.h>
#include <random_quda.h>
#include <mpi_comm_handle.h>

#include <multigrid.h>

#include <deflation.h>


// MUGIQ header files
#include <mugiq.h>


void computeEvecs(void **eVecs_host, double _Complex *eVals_host, QudaEigParam *eigParams){


  // Call the QUDA function
  eigensolveQuda(eVecs_host, eVals_host, eigParams);

}
