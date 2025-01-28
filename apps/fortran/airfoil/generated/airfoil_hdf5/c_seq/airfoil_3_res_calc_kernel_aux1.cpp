#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_3_res_calc_main {

static void res_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    f2c::Ptr<const float> _f2c_ptr_q2,
    const float adt1,
    const float adt2,
    f2c::Ptr<float> _f2c_ptr_res1,
    f2c::Ptr<float> _f2c_ptr_res2
);


static void res_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    f2c::Ptr<const float> _f2c_ptr_q2,
    const float adt1,
    const float adt2,
    f2c::Ptr<float> _f2c_ptr_res1,
    f2c::Ptr<float> _f2c_ptr_res2
) {
    const f2c::Span<const float, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> q1{_f2c_ptr_q1, f2c::Extent{1, 4}};
    const f2c::Span<const float, 1> q2{_f2c_ptr_q2, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res1{_f2c_ptr_res1, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res2{_f2c_ptr_res2, f2c::Extent{1, 4}};
    float dx;
    float dy;
    float mu;
    float ri;
    float p1;
    float vol1;
    float p2;
    float vol2;
    float f;

    dx = x1(1) - x2(1);
    dy = x1(2) - x2(2);
    ri = 1.0 / q1(1);
    p1 = gm1 * (q1(4) - 0.5 * ri * (f2c::pow(q1(2), 2) + f2c::pow(q1(3), 2)));
    vol1 = ri * (q1(2) * dy - q1(3) * dx);
    ri = 1.0 / q2(1);
    p2 = gm1 * (q2(4) - 0.5 * ri * (f2c::pow(q2(2), 2) + f2c::pow(q2(3), 2)));
    vol2 = ri * (q2(2) * dy - q2(3) * dx);
    mu = 0.5 * (adt1 + adt2) * eps;
    f = 0.5 * (vol1 * q1(1) + vol2 * q2(1)) + mu * (q1(1) - q2(1));
    res1(1) = res1(1) + f;
    res2(1) = res2(1) - f;
    f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * q2(2) + p2 * dy) + mu * (q1(2) - q2(2));
    res1(2) = res1(2) + f;
    res2(2) = res2(2) - f;
    f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * q2(3) - p2 * dx) + mu * (q1(3) - q2(3));
    res1(3) = res1(3) + f;
    res2(3) = res2(3) - f;
    f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (q2(4) + p2)) + mu * (q1(4) - q2(4));
    res1(4) = res1(4) + f;
    res2(4) = res2(4) - f;
}

}


extern "C" void op2_k_airfoil_3_res_calc_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7
) {
    int n_args = 8;
    op_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

    op_timing2_enter_kernel("airfoil_3_res_calc", "c_seq", "Indirect");

    op_timing2_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_timing2_next("Computation");



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_timing2_next("MPI Wait");
            op_mpi_wait_all(n_args, args);
            op_timing2_next("Computation");
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;
        int *map1 = arg2.map_data + n * arg2.map->dim;


        op2_m_airfoil_3_res_calc_main::res_calc(
            (double *)arg0.data + map0[1 - 1] * 2,
            (double *)arg1.data + map0[2 - 1] * 2,
            (double *)arg2.data + map1[1 - 1] * 4,
            (double *)arg3.data + map1[2 - 1] * 4,
            ((double *)arg4.data + map1[1 - 1] * 1)[0],
            ((double *)arg5.data + map1[2 - 1] * 1)[0],
            (double *)arg6.data + map1[1 - 1] * 4,
            (double *)arg7.data + map1[2 - 1] * 4
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