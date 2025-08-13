#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_jac_mpi_1_res {

inline void res(const double *A, const float *u, float *du, const float *beta) {
  *du += (float)((*beta) * (*A) * (*u));
}}


void op_par_loop_jac_mpi_1_res(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    int n_args = 4;
    op_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg1.map_data + n * arg1.map->dim;


        op2_m_jac_mpi_1_res::res(
            (double *)arg0.data + n * 3,
            (float *)arg1.data + map0[1] * 2,
            (float *)arg2.data + map0[0] * 3,
            (float *)arg3.data
        );

        if (n == set->size - 1) {
        }
    }

    if (n_exec < set->size) {
    }

    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);


    op_mpi_set_dirtybit(n_args, args);
}