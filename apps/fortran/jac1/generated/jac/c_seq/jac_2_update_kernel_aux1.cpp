#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_jac_2_update_main {

static void update(
    f2c::Ptr<const float> _f2c_ptr_r,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_u_sum,
    f2c::Ptr<float> _f2c_ptr_u_max
);


static void update(
    f2c::Ptr<const float> _f2c_ptr_r,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_u_sum,
    f2c::Ptr<float> _f2c_ptr_u_max
) {
    const f2c::Span<const float, 1> r{_f2c_ptr_r, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> du{_f2c_ptr_du, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u{_f2c_ptr_u, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u_sum{_f2c_ptr_u_sum, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u_max{_f2c_ptr_u_max, f2c::Extent{1, 1}};

    u(1) = u(1) + du(1) + alpha * r(1);
    du(1) = 0.0f;
    u_sum(1) = u_sum(1) + f2c::pow(u(1), 2);
    u_max(1) = f2c::max(u_max(1), u(1));
}

}


extern "C" void op2_k_jac_2_update_main_c(
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

    op_timing2_enter_kernel("jac_2_update", "c_seq", "Direct");

    op_timing2_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_timing2_next("Computation");



    for (int n = 0; n < n_exec; ++n) {


        op2_m_jac_2_update_main::update(
            ((double *)arg0.data + n * 1)[0],
            ((double *)arg1.data + n * 1)[0],
            ((double *)arg2.data + n * 1)[0],
            ((double *)arg3.data)[0],
            ((double *)arg4.data)[0]
        );

    }


    op_timing2_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_timing2_next("MPI Reduce");

    op_mpi_reduce(&arg3, (double *)arg3.data);
    op_mpi_reduce(&arg4, (double *)arg4.data);
    op_timing2_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_timing2_exit();
}