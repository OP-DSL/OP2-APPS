#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_reduction_2_edge_count_main {

static void edge_count(
    f2c::Ptr<float> _f2c_ptr_res,
    f2c::Ptr<int> _f2c_ptr_edge_count_result
);


static void edge_count(
    f2c::Ptr<float> _f2c_ptr_res,
    f2c::Ptr<int> _f2c_ptr_edge_count_result
) {
    const f2c::Span<float, 1> res{_f2c_ptr_res, f2c::Extent{1, 4}};
    const f2c::Span<int, 1> edge_count_result{_f2c_ptr_edge_count_result, f2c::Extent{1, 1}};
    int d;

    for (d = 1; d <= 4; ++d) {
        res(d) = 0.0f;
    }
    edge_count_result = edge_count_result + 1;
}

}


extern "C" void op2_k_reduction_2_edge_count_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    int n_args = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    op_timing2_enter_kernel("reduction_2_edge_count", "c_seq", "Indirect");

    op_timing2_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_timing2_next("Computation");

    int gbl1_temp[1];

    memcpy(gbl1_temp, arg1.data, 1 * sizeof(int));

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_timing2_next("MPI Wait");
            op_mpi_wait_all(n_args, args);
            op_timing2_next("Computation");
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_reduction_2_edge_count_main::edge_count(
            (double *)arg0.data + map0[1 - 1] * 4,
            (gbl1_temp)[0]
        );

        if (n == set->size - 1) {
            memcpy(arg1.data, gbl1_temp, 1 * sizeof(int));
        }
    }

    if (n_exec < set->size) {
        memcpy(arg1.data, gbl1_temp, 1 * sizeof(int));
    }

    op_timing2_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_timing2_next("MPI Reduce");

    op_mpi_reduce(&arg1, (int *)arg1.data);
    op_timing2_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_timing2_exit();
}