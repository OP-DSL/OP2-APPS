#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_aero_mpi_3_init_cg {

inline void init_cg(const double *r, double *c, double *u, double *v, double *p) {
  *c += (*r) * (*r);
  *p = *r;
  *u = 0;
  *v = 0;
}}


void op_par_loop_aero_mpi_3_init_cg(
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


        op2_m_aero_mpi_3_init_cg::init_cg(
            (double *)arg0.data + n * 1,
            (double *)arg1.data,
            (double *)arg2.data + n * 1,
            (double *)arg3.data + n * 1,
            (double *)arg4.data + n * 1
        );

    }


    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_mpi_reduce(&arg1, (double *)arg1.data);

    op_mpi_set_dirtybit(n_args, args);
}