#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_jac_mpi_1_res {

inline void res(const double *A, const double *u, double *du,
                const double *beta, const int *index, const int *idx_ppedge0,
                const int *idx_ppedge1) {
  *du += (*beta) * (*A) * (*u);
  printf("edge %d, nodes %d, %d\n", *index, *idx_ppedge0, *idx_ppedge1);
}}


void op_par_loop_jac_mpi_1_res(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6
) {
    int n_args = 7;
    op_arg args[7];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg1.map_data + n * arg1.map->dim;

        int idx = n;
        int idx5 = map0[0];
        int idx6 = map0[1];

        op2_m_jac_mpi_1_res::res(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + map0[1] * 1,
            (double *)arg2.data + map0[0] * 1,
            (double *)arg3.data,
            idx,
            idx5,
            idx6
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