#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_reduction_mpi_1_res_calc {

inline void res_calc(double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}}


void op_par_loop_reduction_mpi_1_res_calc(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    int n_args = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    int gbl1_temp[1];

    memcpy(gbl1_temp, arg1.data, 1 * sizeof(int));

    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_reduction_mpi_1_res_calc::res_calc(
            (double *)arg0.data + map0[0] * 4,
            gbl1_temp
        );

        if (n == set->size - 1) {
            memcpy(arg1.data, gbl1_temp, 1 * sizeof(int));
        }
    }

    if (n_exec < set->size) {
        memcpy(arg1.data, gbl1_temp, 1 * sizeof(int));
    }

    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_mpi_reduce(&arg1, (int *)arg1.data);

    op_mpi_set_dirtybit(n_args, args);
}