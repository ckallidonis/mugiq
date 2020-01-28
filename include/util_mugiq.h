#ifndef _MUGIQ_UTIL_H
#define _MUGIQ_UTIL_H

#define N_DIM_ 4
#define N_SPIN_ 4
#define N_COLOR_ 3
#define N_GAMMA_ (N_SPIN_ * N_SPIN_)
#define SPINOR_SITE_LEN_ (N_SPIN_ * N_COLOR_)
#define GAUGE_SITE_LEN_ (N_COLOR_ * N_COLOR_)
#define GAMMA_LEN_ (N_SPIN_ * N_SPIN_)

#define SPINOR_SITE_IDX(s,c)  ( (c) + N_COLOR_*(s) )
#define GAUGE_SITE_IDX(c1,c2)  ( (c2) + N_COLOR_*(c1) )

#define GAMMA_GEN_IDX(s,c)  ( SPINOR_SITE_IDX((s),(c)) )
#define GAMMA_COEFF_IDX(s1,c1,s2,c2) ( (SPINOR_SITE_IDX((s2),(c2))) + SPINOR_SITE_LEN_ * (SPINOR_SITE_IDX((s1),(c1))) )

#define MUGIQ_MAX_FINE_VEC 24
#define MUGIQ_MAX_COARSE_VEC 256

#define THREADS_PER_BLOCK 64


#endif // _MUGIQ_UTIL_H
