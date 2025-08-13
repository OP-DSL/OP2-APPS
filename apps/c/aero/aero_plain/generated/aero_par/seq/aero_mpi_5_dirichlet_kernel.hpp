#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_aero_mpi_5_dirichlet {

inline void dirichlet(double *res) { *res = 0.0; }}


void op_par_loop_aero_mpi_5_dirichlet(
    const char* name,
    op_set set,
    op_arg arg0
) {
    int n_args = 1;
    op_arg args[1];

    args[0] = arg0;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_aero_mpi_5_dirichlet::dirichlet(
            (double *)arg0.data + map0[0] * 1
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