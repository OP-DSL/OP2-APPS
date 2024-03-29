//
// auto-generated by op2.py
//

#ifdef _OPENMP
  #include <omp.h>
#endif

// global constants
extern double gam;
extern double gm1;
extern double gm1i;
extern double m2;
extern double wtg1[2];
extern double xi1[2];
extern double Ng1[4];
extern double Ng1_xi[4];
extern double wtg2[4];
extern double Ng2[16];
extern double Ng2_xi[32];
extern double minf;
extern double freq;
extern double kappa;
extern double nmode;
extern double mfan;

// header
#include "op_lib_cpp.h"

#ifndef SKIP_DECL_CONST

void op_decl_const_gam(int dim, char const *type,
                       double *dat){
}

void op_decl_const_gm1(int dim, char const *type,
                       double *dat){
}

void op_decl_const_gm1i(int dim, char const *type,
                       double *dat){
}

void op_decl_const_m2(int dim, char const *type,
                       double *dat){
}

void op_decl_const_wtg1(int dim, char const *type,
                       double *dat){
}

void op_decl_const_xi1(int dim, char const *type,
                       double *dat){
}

void op_decl_const_Ng1(int dim, char const *type,
                       double *dat){
}

void op_decl_const_Ng1_xi(int dim, char const *type,
                       double *dat){
}

void op_decl_const_wtg2(int dim, char const *type,
                       double *dat){
}

void op_decl_const_Ng2(int dim, char const *type,
                       double *dat){
}

void op_decl_const_Ng2_xi(int dim, char const *type,
                       double *dat){
}

void op_decl_const_minf(int dim, char const *type,
                       double *dat){
}

void op_decl_const_freq(int dim, char const *type,
                       double *dat){
}

void op_decl_const_kappa(int dim, char const *type,
                       double *dat){
}

void op_decl_const_nmode(int dim, char const *type,
                       double *dat){
}

void op_decl_const_mfan(int dim, char const *type,
                       double *dat){
}
#endif

// user kernel files
#include "res_calc_kernel.cpp"
#include "dirichlet_kernel.cpp"
#include "init_cg_kernel.cpp"
#include "spMV_kernel.cpp"
#include "dotPV_kernel.cpp"
#include "updateUR_kernel.cpp"
#include "dotR_kernel.cpp"
#include "updateP_kernel.cpp"
#include "update_kernel.cpp"
