extern double alpha;

#ifndef OP_FUN_PREFIX
#define OP_FUN_PREFIX
#endif

static inline OP_FUN_PREFIX double maxfun(double a, double b) {
   return a>b ? a : b;
}


#include "op_lib_cpp.h"

#include "jac_mpi_1_res_kernel.hpp"
#include "jac_mpi_2_update_kernel.hpp"
