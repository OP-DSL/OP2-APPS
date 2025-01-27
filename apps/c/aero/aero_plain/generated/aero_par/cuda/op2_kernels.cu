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

    if (size > MAX_CONST_SIZE) {
        printf("error: requested size %d for const %s exceeds MAX_CONST_SIZE\n", size, name);
        exit(1);
    }

    if (!strcmp(name, "gam")) {
        cutilSafeCall(cudaMemcpyToSymbol(gam_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "gm1")) {
        cutilSafeCall(cudaMemcpyToSymbol(gm1_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "gm1i")) {
        cutilSafeCall(cudaMemcpyToSymbol(gm1i_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "m2")) {
        cutilSafeCall(cudaMemcpyToSymbol(m2_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "wtg1")) {
        cutilSafeCall(cudaMemcpyToSymbol(wtg1_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "xi1")) {
        cutilSafeCall(cudaMemcpyToSymbol(xi1_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng1")) {
        cutilSafeCall(cudaMemcpyToSymbol(Ng1_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng1_xi")) {
        cutilSafeCall(cudaMemcpyToSymbol(Ng1_xi_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "wtg2")) {
        cutilSafeCall(cudaMemcpyToSymbol(wtg2_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng2")) {
        cutilSafeCall(cudaMemcpyToSymbol(Ng2_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "Ng2_xi")) {
        cutilSafeCall(cudaMemcpyToSymbol(Ng2_xi_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "minf")) {
        cutilSafeCall(cudaMemcpyToSymbol(minf_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "freq")) {
        cutilSafeCall(cudaMemcpyToSymbol(freq_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "kappa")) {
        cutilSafeCall(cudaMemcpyToSymbol(kappa_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "nmode")) {
        cutilSafeCall(cudaMemcpyToSymbol(nmode_d, dat, dim * size));
        return;
    }
    if (!strcmp(name, "mfan")) {
        cutilSafeCall(cudaMemcpyToSymbol(mfan_d, dat, dim * size));
        return;
    }

    printf("error: unknown const name %s\n", name);
    exit(1);
}

#include "aero_mpi_1_res_calc_kernel.hpp"
#include "aero_mpi_2_dirichlet_kernel.hpp"
#include "aero_mpi_3_init_cg_kernel.hpp"
#include "aero_mpi_4_spMV_kernel.hpp"
#include "aero_mpi_5_dirichlet_kernel.hpp"
#include "aero_mpi_6_dotPV_kernel.hpp"
#include "aero_mpi_7_updateUR_kernel.hpp"
#include "aero_mpi_8_dotR_kernel.hpp"
#include "aero_mpi_9_updateP_kernel.hpp"
#include "aero_mpi_10_update_kernel.hpp"
