#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_2_adt_calc_main {

static void adt_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_x3,
    f2c::Ptr<const float> _f2c_ptr_x4,
    f2c::Ptr<const float> _f2c_ptr_q,
    float& adt
);


static void adt_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_x3,
    f2c::Ptr<const float> _f2c_ptr_x4,
    f2c::Ptr<const float> _f2c_ptr_q,
    float& adt
) {
    const f2c::Span<const float, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x3{_f2c_ptr_x3, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x4{_f2c_ptr_x4, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    float dx;
    float dy;
    float ri;
    float u;
    float v;
    float c;

    ri = 1.0 / q(1);
    u = ri * q(2);
    v = ri * q(3);
    c = f2c::sqrt(gam * gm1 * (ri * q(4) - 0.5 * (f2c::pow(u, 2) + f2c::pow(v, 2))));
    dx = x2(1) - x1(1);
    dy = x2(2) - x1(2);
    adt = f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x3(1) - x2(1);
    dy = x3(2) - x2(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x4(1) - x3(1);
    dy = x4(2) - x3(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x1(1) - x4(1);
    dy = x1(2) - x4(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    adt = adt / cfl;
}

}


extern "C" void op2_k_airfoil_2_adt_calc_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    int n_args = 6;
    op_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

    op_timing2_enter_kernel("airfoil_2_adt_calc", "c_seq", "Indirect");

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

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_airfoil_2_adt_calc_main::adt_calc(
            (double *)arg0.data + map0[1 - 1] * 2,
            (double *)arg1.data + map0[2 - 1] * 2,
            (double *)arg2.data + map0[3 - 1] * 2,
            (double *)arg3.data + map0[4 - 1] * 2,
            (double *)arg4.data + n * 4,
            ((double *)arg5.data + n * 1)[0]
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