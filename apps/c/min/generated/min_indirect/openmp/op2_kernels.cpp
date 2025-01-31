
#ifdef _OPENMP
#include <omp.h>
#endif

#include "op_lib_cpp.h"

template<typename T, unsigned A = OP2_ALIGNMENT>
inline T *assume_aligned(T *p) {
    return reinterpret_cast<T *>(__builtin_assume_aligned(p, A));
}


#include "min_indirect_1_min_kernel_kernel.hpp"
