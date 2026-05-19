extern double alpha;

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef OP_FUN_PREFIX
#define OP_FUN_PREFIX
#endif

static inline OP_FUN_PREFIX double maxfun(double a, double b) {
   return a>b ? a : b;
}


#include "op_lib_cpp.h"

template<typename T, unsigned A = OP2_ALIGNMENT>
inline T *assume_aligned(T *p) {
    return reinterpret_cast<T *>(__builtin_assume_aligned(p, A));
}


#include "jac_mpi_1_res_kernel.hpp"
#include "jac_mpi_2_update_kernel.hpp"
