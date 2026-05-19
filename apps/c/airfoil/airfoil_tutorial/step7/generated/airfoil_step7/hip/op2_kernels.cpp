__constant__ double gam_d;
__constant__ double gm1_d;
__constant__ double cfl_d;
__constant__ double eps_d;
__constant__ double mach_d;
__constant__ double alpha_d;
__constant__ double qinf_d[4];

#include "op_lib_cpp.h"
#include "op_cuda_rt_support.h"
#include "op_cuda_reduction.h"

#ifndef MAX_CONST_SIZE
#define MAX_CONST_SIZE 128
#endif

void op_decl_const_char(int dim, const char *type, int size, char *dat, const char *name) {
    if (!OP_hybrid_gpu) return;

    if (!strcmp(name, "gam")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(gam_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "gm1")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(gm1_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "cfl")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(cfl_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "eps")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(eps_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "mach")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(mach_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "alpha")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(alpha_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "qinf")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(qinf_d), dat, dim * size));
        return;
    }

    printf("error: unknown const name %s\n", name);
    exit(1);
}

#include "airfoil_step7_1_save_soln_kernel.hpp"
#include "airfoil_step7_2_adt_calc_kernel.hpp"
#include "airfoil_step7_3_res_calc_kernel.hpp"
#include "airfoil_step7_4_bres_calc_kernel.hpp"
#include "airfoil_step7_5_update_kernel.hpp"
