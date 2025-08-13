#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_aero_7_updateUR {

inline void updateUR(double *u, double *r, const double *p, double *v,
                     const double *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}}


void op_par_loop_aero_7_updateUR(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    int n_args = 5;
    op_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {


        op2_m_aero_7_updateUR::updateUR(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + n * 1,
            (double *)arg2.data + n * 1,
            (double *)arg3.data + n * 1,
            (double *)arg4.data
        );

    }


    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);


    op_mpi_set_dirtybit(n_args, args);
}