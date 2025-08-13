#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_4_bres_calc_main {

static void bres_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    const float adt1,
    f2c::Ptr<float> _f2c_ptr_res1,
    const int bound
);


static void bres_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    const float adt1,
    f2c::Ptr<float> _f2c_ptr_res1,
    const int bound
) {
    const f2c::Span<const float, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> q1{_f2c_ptr_q1, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res1{_f2c_ptr_res1, f2c::Extent{1, 4}};
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
    if (bound == 1) {
        res1(2) = res1(2) + p1 * dy;
        res1(3) = res1(3) - p1 * dx;
        return;
    }
    vol1 = ri * (q1(2) * dy - q1(3) * dx);
    ri = 1.0 / qinf[(1) - 1];
    p2 = gm1 * (qinf[(4) - 1] - 0.5 * ri * (f2c::pow(qinf[(2) - 1], 2) + f2c::pow(qinf[(3) - 1], 2)));
    vol2 = ri * (qinf[(2) - 1] * dy - qinf[(3) - 1] * dx);
    mu = adt1 * eps;
    f = 0.5 * (vol1 * q1(1) + vol2 * qinf[(1) - 1]) + mu * (q1(1) - qinf[(1) - 1]);
    res1(1) = res1(1) + f;
    f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * qinf[(2) - 1] + p2 * dy) + mu * (q1(2) - qinf[(2) - 1]);
    res1(2) = res1(2) + f;
    f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * qinf[(3) - 1] - p2 * dx) + mu * (q1(3) - qinf[(3) - 1]);
    res1(3) = res1(3) + f;
    f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (qinf[(4) - 1] + p2)) + mu * (q1(4) - qinf[(4) - 1]);
    res1(4) = res1(4) + f;
}

}


extern "C" void op2_k_airfoil_4_bres_calc_main_c(
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

    op_timing2_enter_kernel("airfoil_4_bres_calc", "c_seq", "Indirect");

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
        int *map1 = arg2.map_data + n * arg2.map->dim;


        op2_m_airfoil_4_bres_calc_main::bres_calc(
            (double *)arg0.data + map0[1 - 1] * 2,
            (double *)arg1.data + map0[2 - 1] * 2,
            (double *)arg2.data + map1[1 - 1] * 4,
            ((double *)arg3.data + map1[1 - 1] * 1)[0],
            (double *)arg4.data + map1[1 - 1] * 4,
            ((int *)arg5.data + n * 1)[0]
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