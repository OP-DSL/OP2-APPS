//
// auto-generated by op2.py
//

//global constants
#ifndef MAX_CONST_SIZE
#define MAX_CONST_SIZE 128
#endif

__constant__ double gam_cuda;
__constant__ double gm1_cuda;
__constant__ double cfl_cuda;
__constant__ double eps_cuda;
__constant__ double mach_cuda;
__constant__ double alpha_cuda;
__constant__ double qinf_cuda[4];

//header
#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

void op_decl_const_gam(int dim, char const *type,
                       double *dat){
  if (!OP_hybrid_gpu) return;
  cutilSafeCall(cudaMemcpyToSymbol(gam_cuda, dat, dim*sizeof(double)));
}

void op_decl_const_gm1(int dim, char const *type,
                       double *dat){
  if (!OP_hybrid_gpu) return;
  cutilSafeCall(cudaMemcpyToSymbol(gm1_cuda, dat, dim*sizeof(double)));
}

void op_decl_const_cfl(int dim, char const *type,
                       double *dat){
  if (!OP_hybrid_gpu) return;
  cutilSafeCall(cudaMemcpyToSymbol(cfl_cuda, dat, dim*sizeof(double)));
}

void op_decl_const_eps(int dim, char const *type,
                       double *dat){
  if (!OP_hybrid_gpu) return;
  cutilSafeCall(cudaMemcpyToSymbol(eps_cuda, dat, dim*sizeof(double)));
}

void op_decl_const_mach(int dim, char const *type,
                       double *dat){
  if (!OP_hybrid_gpu) return;
  cutilSafeCall(cudaMemcpyToSymbol(mach_cuda, dat, dim*sizeof(double)));
}

void op_decl_const_alpha(int dim, char const *type,
                       double *dat){
  if (!OP_hybrid_gpu) return;
  cutilSafeCall(cudaMemcpyToSymbol(alpha_cuda, dat, dim*sizeof(double)));
}

void op_decl_const_qinf(int dim, char const *type,
                       double *dat){
  if (!OP_hybrid_gpu) return;
  if (dim*sizeof(double)>MAX_CONST_SIZE) {
    printf("error: MAX_CONST_SIZE not big enough\n"); exit(1);
  }
  cutilSafeCall(cudaMemcpyToSymbol(qinf_cuda, dat, dim*sizeof(double)));
}

//user kernel files
#include "save_soln_kernel.cu"
#include "adt_calc_kernel.cu"
#include "res_calc_kernel.cu"
#include "bres_calc_kernel.cu"
#include "update_kernel.cu"
