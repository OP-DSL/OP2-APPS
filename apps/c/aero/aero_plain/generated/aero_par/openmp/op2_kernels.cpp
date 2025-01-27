extern double gam;
extern double gm1;
extern double gm1i;
extern double m2;
extern double wtg1[2];
extern double xi1[2];
extern double Ng1[4];
extern double Ng1_xi[4];
extern double wtg2[4];
extern double Ng2[16];
extern double Ng2_xi[32];
extern double minf;
extern double freq;
extern double kappa;
extern double nmode;
extern double mfan;

#ifdef _OPENMP
#include <omp.h>
#endif

#include "op_lib_cpp.h"

template<typename T, unsigned A = OP2_ALIGNMENT>
inline T *assume_aligned(T *p) {
    return reinterpret_cast<T *>(__builtin_assume_aligned(p, A));
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
