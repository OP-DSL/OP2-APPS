__constant__ double gam_d;
__constant__ double gm1_d;
__constant__ double gm1i_d;
__constant__ double m2_d;
__constant__ double wtg1_d[2];
__constant__ double xi1_d[2];
__constant__ double Ng1_d[4];
__constant__ double Ng1_xi_d[4];
__constant__ double wtg2_d[4];
__constant__ double Ng2_d[16];
__constant__ double Ng2_xi_d[32];
__constant__ double minf_d;
__constant__ double freq_d;
__constant__ double kappa_d;
__constant__ double nmode_d;
__constant__ double mfan_d;

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
    if (!strcmp(name, "gm1i")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(gm1i_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "m2")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(m2_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "wtg1")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(wtg1_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "xi1")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(xi1_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng1")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(Ng1_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng1_xi")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(Ng1_xi_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "wtg2")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(wtg2_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng2")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(Ng2_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng2_xi")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(Ng2_xi_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "minf")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(minf_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "freq")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(freq_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "kappa")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(kappa_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "nmode")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(nmode_d), dat, dim * size));
        return;
    }
    if (!strcmp(name, "mfan")) {
        cutilSafeCall(hipMemcpyToSymbol(HIP_SYMBOL(mfan_d), dat, dim * size));
        return;
    }

    printf("error: unknown const name %s\n", name);
    exit(1);
}

#include "aero_1_res_calc_kernel.hpp"
#include "aero_2_dirichlet_kernel.hpp"
#include "aero_3_init_cg_kernel.hpp"
#include "aero_4_spMV_kernel.hpp"
#include "aero_5_dirichlet_kernel.hpp"
#include "aero_6_dotPV_kernel.hpp"
#include "aero_7_updateUR_kernel.hpp"
#include "aero_8_dotR_kernel.hpp"
#include "aero_9_updateP_kernel.hpp"
#include "aero_10_update_kernel.hpp"
