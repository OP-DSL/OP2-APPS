extern double gam;
extern double gm1;
extern double cfl;
extern double eps;
extern double mach;
extern double alpha;
extern double qinf[4];

#ifdef _OPENMP
#include <omp.h>
#endif

#include "op_lib_cpp.h"

template<typename T, unsigned A = OP2_ALIGNMENT>
inline T *assume_aligned(T *p) {
    return reinterpret_cast<T *>(__builtin_assume_aligned(p, A));
}


#include "airfoil_mpi_1_save_soln_kernel.hpp"
#include "airfoil_mpi_2_adt_calc_kernel.hpp"
#include "airfoil_mpi_3_res_calc_kernel.hpp"
#include "airfoil_mpi_4_bres_calc_kernel.hpp"
#include "airfoil_mpi_5_update_kernel.hpp"
