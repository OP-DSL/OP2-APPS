#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_jac_1_res_m {

static void res(
    f2c::Ptr<const float> _f2c_ptr_a,
    f2c::Ptr<const float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<const float> _f2c_ptr_beta
);


static void res(
    f2c::Ptr<const float> _f2c_ptr_a,
    f2c::Ptr<const float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<const float> _f2c_ptr_beta
) {
    const f2c::Span<const float, 1> a{_f2c_ptr_a, f2c::Extent{1, 1}};
    const f2c::Span<const float, 1> u{_f2c_ptr_u, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> du{_f2c_ptr_du, f2c::Extent{1, 1}};
    const f2c::Span<const float, 1> beta{_f2c_ptr_beta, f2c::Extent{1, 1}};

    du(1) = du(1) + beta(1) * a(1) * u(1);
}

}


extern "C" void op2_k_jac_1_res_m_c(
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

    op_timing2_enter_kernel("jac_1_res", "c_seq", "Indirect");

    op_timing2_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_timing2_next("Computation");



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

        int *map0 = arg1.map_data + n * arg1.map->dim;


        op2_m_jac_1_res_m::res(
            ((double *)arg0.data + n * 1)[0],
            ((double *)arg1.data + map0[2 - 1] * 1)[0],
            ((double *)arg2.data + map0[1 - 1] * 1)[0],
            ((double *)arg3.data)[0]
        );

        if (n == set->size - 1) {
        }
    }

    if (n_exec < set->size) {
    }

    op_timing2_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_timing2_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_timing2_exit();
}