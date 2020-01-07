#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <util_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <test_params_mugiq.h>
#include "misc.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#include <mugiq.h>

double kappa5; // Derived, not given. Used in matVec checks.

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  if(mugiq_eig_operator == MUGIQ_EIG_OPERATOR_MG){
    printfQuda("MG parameters\n");
    printfQuda(" - number of levels %d\n", mg_levels);
    for (int i=0; i<mg_levels-1; i++) {
      printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
      printfQuda(" - level %d number of pre-smoother applications %d\n", i+1, nu_pre[i]);
      printfQuda(" - level %d number of post-smoother applications %d\n", i+1, nu_post[i]);
    }
  }
    
  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
  printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
  printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
  printfQuda(" - size of Krylov space %d\n", eig_nKr);
  printfQuda(" - solver tolerance %e\n", eig_tol);
  printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
  if (eig_compute_svd) {
    printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
    printfQuda(" - ***********************************************************\n");
    printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
    printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
    printfQuda(" - ***********************************************************\n");
  } else {
    printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
               eig_use_normop ? "true" : "false");
  }
  if (eig_use_poly_acc) {
    printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
    printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
    printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
  }
  else {
    printfQuda(" Not using Chebyshev acceleration\n");
  }
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}
//-------------------------------------------------------------------------------


//- Precision settings
QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;
QudaPrecision &cuda_prec_refinement_sloppy = prec_refinement_sloppy;
//-------------------------------------------------------------------------------


//- Set the Gauge field Parameters
void setGaugeParam(QudaGaugeParam &gauge_param)
{
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = (cuda_prec_precondition == QUDA_INVALID_PRECISION) ? cuda_prec_sloppy : cuda_prec_precondition;
  gauge_param.cuda_prec_refinement_sloppy = (cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION) ? cuda_prec_sloppy : cuda_prec_refinement_sloppy;
  gauge_param.reconstruct_precondition = link_recon_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;
  
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int y_face_size = gauge_param.X[0] * gauge_param.X[2] * gauge_param.X[3] / 2;
  int z_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[3] / 2;
  int t_face_size = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] / 2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;
#endif
}
//-------------------------------------------------------------------------------


//-C.K. Set Inverter Parameters
void setInvertParam(QudaInvertParam &inv_param)
{

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass));
    if (dslash_type == QUDA_LAPLACE_DSLASH) inv_param.kappa = 1.0 / (8 + mass);
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5 / kappa - (1.0 + 3.0 / anisotropy);
    if (dslash_type == QUDA_LAPLACE_DSLASH) inv_param.mass = 1.0 / kappa - 8.0;
  }
  inv_param.laplace3D = laplace3D;

  printfQuda("Kappa = %.8f Mass = %.8f\n", inv_param.kappa, inv_param.mass);

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = (cuda_prec_precondition == QUDA_INVALID_PRECISION) ? cuda_prec_sloppy : cuda_prec_precondition;
  inv_param.cuda_prec_refinement_sloppy = (cuda_prec_refinement_sloppy == QUDA_INVALID_PRECISION) ? cuda_prec_sloppy : cuda_prec_refinement_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;


  
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {

    inv_param.mu = mu;
    inv_param.epsilon = 0.1385;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
             || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    inv_param.m5 = m5;
    kappa5 = 0.5 / (5 + inv_param.m5);
    inv_param.Ls = Lsdim;
    for (int k = 0; k < Lsdim; k++) { // for mobius only
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = b5;
      inv_param.c_5[k] = c5;
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.solution_type = solution_type;
  inv_param.solve_type = (inv_param.solution_type == QUDA_MAT_SOLUTION ? QUDA_DIRECT_SOLVE : QUDA_DIRECT_PC_SOLVE);

  inv_param.matpc_type = matpc_type;
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.inv_type = inv_type;
  inv_param.verbosity = verbosity;
  inv_param.inv_type_precondition = precon_type;
  inv_param.tol = tol;

  // require both L2 relative and heavy quark residual to determine
  // convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // specify a tolerance for the residual for heavy quark residual
  inv_param.tol_hq = tol_hq;

  // these can be set individually
  for (int i = 0; i < inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.gcrNkrylov = 10;
  inv_param.pipeline = 10;
  inv_param.reliable_delta = 1e-4;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}
//-------------------------------------------------------------------------------


// Parameters defining the eigensolver
void setEigParam(QudaEigParam &eig_param)
{
  eig_param.eig_type = eig_type;
  eig_param.spectrum = eig_spectrum;
  if ((eig_type == QUDA_EIG_TR_LANCZOS || eig_type == QUDA_EIG_IR_LANCZOS)
      && !(eig_spectrum == QUDA_SPECTRUM_LR_EIG || eig_spectrum == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to Lanczos type solver");
  }

  // The solver will exit when nConv extremal eigenpairs have converged
  if (eig_nConv < 0) {
    eig_param.nConv = eig_nEv;
    eig_nConv = eig_nEv;
  } else {
    eig_param.nConv = eig_nConv;
  }

  eig_param.nEv = eig_nEv;
  eig_param.nKr = eig_nKr;
  eig_param.tol = eig_tol;
  eig_param.batched_rotate = eig_batched_rotate;
  eig_param.require_convergence = eig_require_convergence ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.check_interval = eig_check_interval;
  eig_param.max_restarts = eig_max_restarts;
  eig_param.cuda_prec_ritz = cuda_prec;

  eig_param.use_norm_op = eig_use_normop ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.use_dagger = eig_use_dagger ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.compute_svd = eig_compute_svd ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  if (eig_compute_svd) {
    eig_param.use_dagger = QUDA_BOOLEAN_NO;
    eig_param.use_norm_op = QUDA_BOOLEAN_YES;
  }

  eig_param.use_poly_acc = eig_use_poly_acc ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.poly_deg = eig_poly_deg;
  eig_param.a_min = eig_amin;
  eig_param.a_max = eig_amax;

  //-Override use_poly-acc option when using PRIMME
  if(eig_type == QUDA_EIG_PRIMME) eig_param.use_poly_acc = QUDA_BOOLEAN_NO;
  
  eig_param.arpack_check = eig_arpack_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  strcpy(eig_param.arpack_logfile, eig_arpack_logfile);
  strcpy(eig_param.QUDA_logfile, eig_QUDA_logfile);

  strcpy(eig_param.vec_infile, eig_vec_infile);
  strcpy(eig_param.vec_outfile, eig_vec_outfile);
}
//-------------------------------------------------------------------------------


// Set eigensolver Parameters for MG-eigensolver
void setEigParamMG(QudaEigParam &mg_eig_param, int level)
{
  mg_eig_param.eig_type = mg_eig_type[level];
  mg_eig_param.spectrum = mg_eig_spectrum[level];
  if ((mg_eig_type[level] == QUDA_EIG_TR_LANCZOS || mg_eig_type[level] == QUDA_EIG_IR_LANCZOS)
      && !(mg_eig_spectrum[level] == QUDA_SPECTRUM_LR_EIG || mg_eig_spectrum[level] == QUDA_SPECTRUM_SR_EIG)) {
    errorQuda("Only real spectrum type (LR or SR) can be passed to the a Lanczos type solver");
  }

  mg_eig_param.nEv = mg_eig_nEv[level];
  mg_eig_param.nKr = mg_eig_nKr[level];
  mg_eig_param.nConv = nvec[level];
  mg_eig_param.batched_rotate = mg_eig_batched_rotate[level];
  mg_eig_param.require_convergence = mg_eig_require_convergence[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  mg_eig_param.tol = mg_eig_tol[level];
  mg_eig_param.check_interval = mg_eig_check_interval[level];
  mg_eig_param.max_restarts = mg_eig_max_restarts[level];
  mg_eig_param.cuda_prec_ritz = cuda_prec;

  mg_eig_param.compute_svd = QUDA_BOOLEAN_NO;
  mg_eig_param.use_norm_op = mg_eig_use_normop[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_eig_param.use_dagger = mg_eig_use_dagger[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  mg_eig_param.use_poly_acc = mg_eig_use_poly_acc[level] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_eig_param.poly_deg = mg_eig_poly_deg[level];
  mg_eig_param.a_min = mg_eig_amin[level];
  mg_eig_param.a_max = mg_eig_amax[level];

  // set file i/o parameters
  // Give empty strings, Multigrid will handle IO.
  strcpy(mg_eig_param.vec_infile, "");
  strcpy(mg_eig_param.vec_outfile, "");

  strcpy(mg_eig_param.QUDA_logfile, eig_QUDA_logfile);
}
//-------------------------------------------------------------------------------


//- Set default MG parameters
void setDefaultMGParams(){

  solve_type = QUDA_DIRECT_SOLVE;
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_VERBOSE;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    setup_tol[i] = 5e-6;
    setup_maxiter[i] = 500;
    mu_factor[i] = 1.;
    coarse_solve_type[i] = QUDA_INVALID_SOLVE;
    smoother_solve_type[i] = QUDA_INVALID_SOLVE;
    schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_MR_INVERTER;
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 100;
    solver_location[i] = QUDA_CUDA_FIELD_LOCATION;
    setup_location[i] = QUDA_CUDA_FIELD_LOCATION;
    nu_pre[i] = 2;
    nu_post[i] = 2;
    n_block_ortho[i] = 1;

    // Default eigensolver params
    mg_eig[i] = false;
    mg_eig_tol[i] = 1e-3;
    mg_eig_require_convergence[i] = QUDA_BOOLEAN_YES;
    mg_eig_type[i] = QUDA_EIG_TR_LANCZOS;
    mg_eig_spectrum[i] = QUDA_SPECTRUM_SR_EIG;
    mg_eig_check_interval[i] = 5;
    mg_eig_max_restarts[i] = 100;
    mg_eig_use_normop[i] = QUDA_BOOLEAN_NO;
    mg_eig_use_dagger[i] = QUDA_BOOLEAN_NO;
    mg_eig_use_poly_acc[i] = QUDA_BOOLEAN_YES;
    mg_eig_poly_deg[i] = 100;
    mg_eig_amin[i] = 1.0;
    mg_eig_amax[i] = 5.0;

    setup_ca_basis[i] = QUDA_POWER_BASIS;
    setup_ca_basis_size[i] = 4;
    setup_ca_lambda_min[i] = 0.0;
    setup_ca_lambda_max[i] = -1.0; // use power iterations

    coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
    coarse_solver_ca_basis_size[i] = 4;
    coarse_solver_ca_lambda_min[i] = 0.0;
    coarse_solver_ca_lambda_max[i] = -1.0;

    strcpy(mg_vec_infile[i], "");
    strcpy(mg_vec_outfile[i], "");
  }
  reliable_delta = 1e-4;

  for (int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    if (coarse_solve_type[i] == QUDA_INVALID_SOLVE) coarse_solve_type[i] = solve_type;
    if (smoother_solve_type[i] == QUDA_INVALID_SOLVE) smoother_solve_type[i] = solve_type;//QUDA_DIRECT_PC_SOLVE;
  }

}
//-------------------------------------------------------------------------------

//- Set MultiGrid Parameters
void setMultigridParam(QudaMultigridParam &mg_param)
{
  QudaInvertParam &inv_param = *mg_param.invert_param;

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec_sloppy;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1 + 3/anisotropy);
  }

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;
    
    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j = 0; j < 4; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    for (int j = 4; j < QUDA_MAX_DIM; j++) mg_param.geo_block_size[i][j] = 1;
    mg_param.use_eig_solver[i] = mg_eig[i] ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
    mg_param.verbosity[i] = mg_verbosity[i];
    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];
    mg_param.setup_tol[i] = setup_tol[i];
    mg_param.setup_maxiter[i] = setup_maxiter[i];

    // Basis to use for CA-CGN(E/R) setup
    mg_param.setup_ca_basis[i] = setup_ca_basis[i];

    // Basis size for CACG setup
    mg_param.setup_ca_basis_size[i] = setup_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis setup
    mg_param.setup_ca_lambda_min[i] = setup_ca_lambda_min[i];
    mg_param.setup_ca_lambda_max[i] = setup_ca_lambda_max[i];

        mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i]; // default to 24 vectors if not set
    mg_param.n_block_ortho[i] = n_block_ortho[i];    // number of times to Gram-Schmidt
    mg_param.precision_null[i] = prec_null; // precision to store the null-space basis
    mg_param.smoother_halo_precision[i] = smoother_halo_prec; // precision of the halo exchange in the smoother
    mg_param.nu_pre[i] = nu_pre[i];
    mg_param.nu_post[i] = nu_post[i];
    mg_param.mu_factor[i] = mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];

    // Basis to use for CA-CGN(E/R) coarse solver
    mg_param.coarse_solver_ca_basis[i] = coarse_solver_ca_basis[i];

    // Basis size for CACG coarse solver/
    mg_param.coarse_solver_ca_basis_size[i] = coarse_solver_ca_basis_size[i];

    // Minimum and maximum eigenvalue for Chebyshev CA basis
    mg_param.coarse_solver_ca_lambda_min[i] = coarse_solver_ca_lambda_min[i];
    mg_param.coarse_solver_ca_lambda_max[i] = coarse_solver_ca_lambda_max[i];

    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i];

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    mg_param.smoother_solve_type[i] = smoother_solve_type[i];

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = schwarz_cycle[i];

    if (i == 0) { // top-level treatment
      if (coarse_solve_type[0] != solve_type)
        errorQuda("Mismatch between top-level MG solve type %d and outer solve type %d", coarse_solve_type[0], solve_type);

      if (solve_type == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", solve_type);
      }

    } else {

      if (coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", coarse_solve_type[i]);
      }

    }

    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = solver_location[i];
    mg_param.setup_location[i] = setup_location[i];
  }

  // whether to run GPU setup but putting temporaries into mapped (slow CPU) memory
  mg_param.setup_minimize_memory = QUDA_BOOLEAN_NO;

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2;

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES
    : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_param.run_low_mode_check = low_mode_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  mg_param.run_oblique_proj_check = oblique_proj_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  // set file i/o parameters
  for (int i = 0; i < mg_param.n_level; i++) {
    strcpy(mg_param.vec_infile[i], mg_vec_infile[i]);
    strcpy(mg_param.vec_outfile[i], mg_vec_outfile[i]);
    if (strcmp(mg_param.vec_infile[i], "") != 0) mg_param.vec_load[i] = QUDA_BOOLEAN_YES;
    if (strcmp(mg_param.vec_outfile[i], "") != 0) mg_param.vec_store[i] = QUDA_BOOLEAN_YES;
  }

  mg_param.coarse_guess = mg_eig_coarse_guess ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 10;

  inv_param.verbosity = verbosity;
  inv_param.verbosity_precondition = verbosity;
}
//-------------------------------------------------------------------------------


int main(int argc, char **argv)
{
  // Parse QUDA and MuGiq command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_eigen_option_mugiq(app);
  
  //- Parse MG parameters
  setDefaultMGParams(); //- Det default values before parsing
  add_multigrid_option_group(app);
  
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  //- Set some Precision defaults if not parsed
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (smoother_halo_prec == QUDA_INVALID_PRECISION) smoother_halo_prec = prec_null;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  
  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  // QUDA parameters begin here.
  //------------------------------------------------------------------------------
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);
  
  QudaEigParam eig_param = newQudaEigParam();
  QudaInvertParam eig_inv_param = newQudaInvertParam();
  setInvertParam(eig_inv_param);
  eig_param.invert_param = &eig_inv_param;
  setEigParam(eig_param);

  if (eig_param.arpack_check)
    errorQuda("MuGiq does not support ARPACK!\n");

  QudaMultigridParam mg_param;
  QudaInvertParam mg_inv_param;
  QudaEigParam mg_eig_param[mg_levels];
  if(mugiq_eig_operator == MUGIQ_EIG_OPERATOR_MG){
    printfQuda("Setting MG parameters\n");
    mg_param  = newQudaMultigridParam();
    mg_inv_param = newQudaInvertParam();

    for (int i = 0; i < mg_levels; i++) {
      if (mg_eig[i]) {
	mg_eig_param[i] = newQudaEigParam();
	setEigParamMG(mg_eig_param[i], i);
	mg_param.eig_param[i] = &mg_eig_param[i];
      }
      else mg_param.eig_param[i] = nullptr;
    }
    //- Set MG parameters
    mg_param.invert_param = &mg_inv_param;
    setMultigridParam(mg_param);
  }
  
  // All user inputs now defined
  display_test_info();
  
  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    dw_setDims(gauge_param.X, eig_inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  // set spinor site size
  int sss = 24;
  if (dslash_type == QUDA_LAPLACE_DSLASH) sss = 6;
  setSpinorSiteSize(sss);

  // Load the gauge field
  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover = 0, *clover_inv = 0;

  for (int dir = 0; dir < 4; dir++) { gauge[dir] = malloc(V * gaugeSiteSize * gSize); }

  if (strcmp(latfile, "")) { // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate an SU(3) field
    if (unit_gauge) {
      construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    } else {
      construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    }
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are rands in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = eig_inv_param.clover_cpu_prec;
    clover = malloc(V * cloverSiteSize * cSize);
    clover_inv = malloc(V * cloverSiteSize * cSize);
    if (!compute_clover){
      warningQuda("*** Will create RANDOM Clover field ***\n");
      warningQuda("*** If this is not intended behavior set option '--compute-clover true' ***\n");
      construct_clover_field(clover, norm, diag, eig_inv_param.clover_cpu_prec);
    }
    eig_inv_param.compute_clover = compute_clover;
    if (compute_clover) eig_inv_param.return_clover = 1;
    eig_inv_param.compute_clover_inverse = 1;
    eig_inv_param.return_clover_inverse = 1;
  }

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void *)gauge, &gauge_param);
  
  // this line ensure that if we need to construct the clover inverse
  // (in either the smoother or the solver) we do so
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {    
    printfQuda("Loading Clover term\n");
    if((mugiq_eig_operator == MUGIQ_EIG_OPERATOR_MG) &&
       (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE ||
    	solve_type == QUDA_DIRECT_PC_SOLVE)) eig_inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;    
    loadCloverQuda(clover, clover_inv, &eig_inv_param);
  }
  eig_inv_param.solve_type = (eig_inv_param.solution_type == QUDA_MAT_SOLUTION ? QUDA_DIRECT_SOLVE : QUDA_DIRECT_PC_SOLVE);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computing the plaquette...\n");
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // Call the Eigensolver interface-function
  double time = -((double)clock());
  if(mugiq_eig_operator == MUGIQ_EIG_OPERATOR_NO_MG){
    if(mugiq_eig_task == MUGIQ_COMPUTE_EVECS_QUDA){
      void **host_evecs = (void **)malloc(eig_nConv * sizeof(void *));
      for (int i = 0; i < eig_nConv; i++) {
	host_evecs[i] = (void *)malloc(V * eig_inv_param.Ls * sss * eig_inv_param.cpu_prec);
      }
      double _Complex *host_evals = (double _Complex *)malloc(eig_param.nEv * sizeof(double _Complex));
      
      computeEvecsQudaWrapper(host_evecs, host_evals, &eig_param);
      
      for (int i = 0; i < eig_nConv; i++) free(host_evecs[i]);
      free(host_evecs);
      free(host_evals);    
    }
    else if(mugiq_eig_task == MUGIQ_COMPUTE_EVECS_MUGIQ) computeEvecsMuGiq(&eig_param);    
    else errorQuda("Option --mugiq-eig-task not set! (options are computeEvecsQuda,computeEvecsMuGiq)\n");
  }
  else if(mugiq_eig_operator == MUGIQ_EIG_OPERATOR_MG) computeEvecsMuGiq_MG(mg_param, eig_param); //- Compute Coarse MG operator eigenvalues
  else errorQuda("Option --mugiq-eig-operator not set! (options are mg/no_mg)\n");
    
  time += (double)clock();
  printfQuda("Time for solution = %f\n", time / CLOCKS_PER_SEC);
  //----------------------------------------------------------------------------
  

  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) { freeCloverQuda(); }

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);

  return 0;
}
