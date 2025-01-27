
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


    printf("error: unknown const name %s\n", name);
    exit(1);
}

#include "reduction_1_res_calc_kernel.hpp"
#include "reduction_2_update_kernel.hpp"
