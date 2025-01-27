__constant__ double alpha_d;

#ifndef OP_FUN_PREFIX
#define OP_FUN_PREFIX __host__ __device__
#endif

#ifndef OP_FUN_PREFIX
#define OP_FUN_PREFIX
#endif

static inline OP_FUN_PREFIX double maxfun(double a, double b) {
   return a>b ? a : b;
}


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

    if (!strcmp(name, "alpha")) {
        cutilSafeCall(cudaMemcpyToSymbol(alpha_d, dat, dim * size));
        return;
    }

    printf("error: unknown const name %s\n", name);
    exit(1);
}

#include "jac_mpi_1_res_kernel.hpp"
#include "jac_mpi_2_update_kernel.hpp"
