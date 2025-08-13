#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_min_direct_1_min_kernel {

void min_kernel(const int *d, int *min) {
    *min = std::min(*d, *min);
}}


void op_par_loop_min_direct_1_min_kernel(
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



    for (int n = 0; n < n_exec; ++n) {


        op2_m_min_direct_1_min_kernel::min_kernel(
            (int *)arg0.data + n * 1,
            (int *)arg1.data
        );

    }


    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_mpi_reduce(&arg1, (int *)arg1.data);

    op_mpi_set_dirtybit(n_args, args);
}