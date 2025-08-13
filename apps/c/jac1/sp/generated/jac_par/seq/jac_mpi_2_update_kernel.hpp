#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_jac_mpi_2_update {

inline void update(const float *r, float *du, float *u, float *u_sum,
                   float *u_max) {
  *u += *du + alpha * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = ((*u_max > *u) ? (*u_max) : (*u));
}}


void op_par_loop_jac_mpi_2_update(
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


        op2_m_jac_mpi_2_update::update(
            (float *)arg0.data + n * 1,
            (float *)arg1.data + n * 1,
            (float *)arg2.data + n * 1,
            (float *)arg3.data,
            (float *)arg4.data
        );

    }


    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_mpi_reduce(&arg3, (float *)arg3.data);
    op_mpi_reduce(&arg4, (float *)arg4.data);

    op_mpi_set_dirtybit(n_args, args);
}