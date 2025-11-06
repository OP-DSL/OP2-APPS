__constant__ float gam_d;
__constant__ float gm1_d;
__constant__ float cfl_d;
__constant__ float eps_d;
__constant__ float mach_d;
__constant__ float alpha_d;
__constant__ float qinf_d[4];

#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#ifndef MAX_CONST_SIZE
#define MAX_CONST_SIZE 128
#endif

void op_decl_const_char(int dim, const char *type, int size, char *dat, const char *name) {
    if (!OP_hybrid_gpu) return;

    if (!strcmp(name, "gam")) {
        cutilSafeCall(cudaMemcpyToSymbol(gam_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "gm1")) {
        cutilSafeCall(cudaMemcpyToSymbol(gm1_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "cfl")) {
        cutilSafeCall(cudaMemcpyToSymbol(cfl_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "eps")) {
        cutilSafeCall(cudaMemcpyToSymbol(eps_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "mach")) {
        cutilSafeCall(cudaMemcpyToSymbol(mach_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "alpha")) {
        cutilSafeCall(cudaMemcpyToSymbol(alpha_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "qinf")) {
        cutilSafeCall(cudaMemcpyToSymbol(qinf_d, dat, dim * size));
        return;
    }

    printf("error: unknown const name %s\n", name);
    exit(1);
}

#include "airfoil_1_save_soln_kernel.hpp"
#include "airfoil_2_adt_calc_kernel.hpp"
#include "airfoil_3_res_calc_kernel.hpp"
#include "airfoil_4_bres_calc_kernel.hpp"
#include "airfoil_5_update_kernel.hpp"
