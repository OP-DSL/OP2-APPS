#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_1_save_soln_m {

static void save_soln(
    f2c::Ptr<const double> _f2c_ptr_q,
    f2c::Ptr<double> _f2c_ptr_qold
);


static void save_soln(
    f2c::Ptr<const double> _f2c_ptr_q,
    f2c::Ptr<double> _f2c_ptr_qold
) {
    const f2c::Span<const double, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    const f2c::Span<double, 1> qold{_f2c_ptr_qold, f2c::Extent{1, 4}};
    int i;

    for (i = 1; i <= 4; ++i) {
        qold(i) = q(i);
    }
}

}


extern "C" void op2_k_airfoil_1_save_soln_m_c(
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    int n_args = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    op_profile_enter_kernel("airfoil_1_save_soln", "c_seq", "Direct");

    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_profile_next("Computation");



    [[maybe_unused]] int zero_int = 0;
    [[maybe_unused]] bool zero_bool = 0;
    [[maybe_unused]] float zero_float = 0;
    [[maybe_unused]] double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {


        op2_m_airfoil_1_save_soln_m::save_soln(
            (double *)arg0.data + n * 4,
            (double *)arg1.data + n * 4
        );

    }


    op_profile_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_profile_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_profile_exit();
}
#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_2_adt_calc_m {

static void adt_calc(
    f2c::Ptr<const double> _f2c_ptr_x1,
    f2c::Ptr<const double> _f2c_ptr_x2,
    f2c::Ptr<const double> _f2c_ptr_x3,
    f2c::Ptr<const double> _f2c_ptr_x4,
    f2c::Ptr<const double> _f2c_ptr_q,
    double& adt
);


static void adt_calc(
    f2c::Ptr<const double> _f2c_ptr_x1,
    f2c::Ptr<const double> _f2c_ptr_x2,
    f2c::Ptr<const double> _f2c_ptr_x3,
    f2c::Ptr<const double> _f2c_ptr_x4,
    f2c::Ptr<const double> _f2c_ptr_q,
    double& adt
) {
    const f2c::Span<const double, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> x3{_f2c_ptr_x3, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> x4{_f2c_ptr_x4, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    double dx;
    double dy;
    double ri;
    double u;
    double v;
    double c;

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


extern "C" void op2_k_airfoil_2_adt_calc_m_c(
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

    op_profile_enter_kernel("airfoil_2_adt_calc", "c_seq", "Indirect");

    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_profile_next("Computation");



    [[maybe_unused]] int zero_int = 0;
    [[maybe_unused]] bool zero_bool = 0;
    [[maybe_unused]] float zero_float = 0;
    [[maybe_unused]] double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(n_args, args);
            op_profile_next("Computation");
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_airfoil_2_adt_calc_m::adt_calc(
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

    op_profile_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_profile_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_profile_exit();
}
#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_3_res_calc_m {

static void res_calc(
    f2c::Ptr<const double> _f2c_ptr_x1,
    f2c::Ptr<const double> _f2c_ptr_x2,
    f2c::Ptr<const double> _f2c_ptr_q1,
    f2c::Ptr<const double> _f2c_ptr_q2,
    const double adt1,
    const double adt2,
    f2c::Ptr<double> _f2c_ptr_res1,
    f2c::Ptr<double> _f2c_ptr_res2
);


static void res_calc(
    f2c::Ptr<const double> _f2c_ptr_x1,
    f2c::Ptr<const double> _f2c_ptr_x2,
    f2c::Ptr<const double> _f2c_ptr_q1,
    f2c::Ptr<const double> _f2c_ptr_q2,
    const double adt1,
    const double adt2,
    f2c::Ptr<double> _f2c_ptr_res1,
    f2c::Ptr<double> _f2c_ptr_res2
) {
    const f2c::Span<const double, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> q1{_f2c_ptr_q1, f2c::Extent{1, 4}};
    const f2c::Span<const double, 1> q2{_f2c_ptr_q2, f2c::Extent{1, 4}};
    const f2c::Span<double, 1> res1{_f2c_ptr_res1, f2c::Extent{1, 4}};
    const f2c::Span<double, 1> res2{_f2c_ptr_res2, f2c::Extent{1, 4}};
    double dx;
    double dy;
    double mu;
    double ri;
    double p1;
    double vol1;
    double p2;
    double vol2;
    double f;

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


extern "C" void op2_k_airfoil_3_res_calc_m_c(
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

    op_profile_enter_kernel("airfoil_3_res_calc", "c_seq", "Indirect");

    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_profile_next("Computation");



    [[maybe_unused]] int zero_int = 0;
    [[maybe_unused]] bool zero_bool = 0;
    [[maybe_unused]] float zero_float = 0;
    [[maybe_unused]] double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(n_args, args);
            op_profile_next("Computation");
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;
        int *map1 = arg2.map_data + n * arg2.map->dim;


        op2_m_airfoil_3_res_calc_m::res_calc(
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

    op_profile_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_profile_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_profile_exit();
}
#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_4_bres_calc_m {

static void bres_calc(
    f2c::Ptr<const double> _f2c_ptr_x1,
    f2c::Ptr<const double> _f2c_ptr_x2,
    f2c::Ptr<const double> _f2c_ptr_q1,
    const double adt1,
    f2c::Ptr<double> _f2c_ptr_res1,
    const int bound
);


static void bres_calc(
    f2c::Ptr<const double> _f2c_ptr_x1,
    f2c::Ptr<const double> _f2c_ptr_x2,
    f2c::Ptr<const double> _f2c_ptr_q1,
    const double adt1,
    f2c::Ptr<double> _f2c_ptr_res1,
    const int bound
) {
    const f2c::Span<const double, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const double, 1> q1{_f2c_ptr_q1, f2c::Extent{1, 4}};
    const f2c::Span<double, 1> res1{_f2c_ptr_res1, f2c::Extent{1, 4}};
    double dx;
    double dy;
    double mu;
    double ri;
    double p1;
    double vol1;
    double p2;
    double vol2;
    double f;

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


extern "C" void op2_k_airfoil_4_bres_calc_m_c(
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

    op_profile_enter_kernel("airfoil_4_bres_calc", "c_seq", "Indirect");

    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_profile_next("Computation");



    [[maybe_unused]] int zero_int = 0;
    [[maybe_unused]] bool zero_bool = 0;
    [[maybe_unused]] float zero_float = 0;
    [[maybe_unused]] double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(n_args, args);
            op_profile_next("Computation");
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;
        int *map1 = arg2.map_data + n * arg2.map->dim;


        op2_m_airfoil_4_bres_calc_m::bres_calc(
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

    op_profile_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_profile_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_profile_exit();
}
#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_5_update_m {

static void update(
    f2c::Ptr<const double> _f2c_ptr_qold,
    f2c::Ptr<double> _f2c_ptr_q,
    f2c::Ptr<double> _f2c_ptr_res,
    const double adt,
    f2c::Ptr<double> _f2c_ptr_rms,
    double& maxerr,
    const int idx,
    int& errloc
);


static void update(
    f2c::Ptr<const double> _f2c_ptr_qold,
    f2c::Ptr<double> _f2c_ptr_q,
    f2c::Ptr<double> _f2c_ptr_res,
    const double adt,
    f2c::Ptr<double> _f2c_ptr_rms,
    double& maxerr,
    const int idx,
    int& errloc
) {
    const f2c::Span<const double, 1> qold{_f2c_ptr_qold, f2c::Extent{1, 4}};
    const f2c::Span<double, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    const f2c::Span<double, 1> res{_f2c_ptr_res, f2c::Extent{1, 4}};
    const f2c::Span<double, 1> rms{_f2c_ptr_rms, f2c::Extent{1, 2}};
    double del;
    double adti;
    int i;

    adti = 1.0 / adt;
    for (i = 1; i <= 4; ++i) {
        del = adti * res(i);
        q(i) = qold(i) - del;
        res(i) = 0.0;
        rms(2) = rms(2) + f2c::pow(del, 2);
        if (f2c::pow(del, 2) > maxerr) {
            maxerr = f2c::pow(del, 2);
            errloc = idx;
        }
    }
}

}


extern "C" void op2_k_airfoil_5_update_m_c(
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

    op_profile_enter_kernel("airfoil_5_update", "c_seq", "Direct");

    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_profile_next("Computation");



    [[maybe_unused]] int zero_int = 0;
    [[maybe_unused]] bool zero_bool = 0;
    [[maybe_unused]] float zero_float = 0;
    [[maybe_unused]] double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {

        int idx = n + 1;

        op2_m_airfoil_5_update_m::update(
            (double *)arg0.data + n * 4,
            (double *)arg1.data + n * 4,
            (double *)arg2.data + n * 4,
            ((double *)arg3.data + n * 1)[0],
            (double *)arg4.data,
            ((double *)arg5.data)[0],
            idx,
            ((int *)arg7.data)[0]
        );

    }


    op_profile_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_profile_next("MPI Reduce");

    op_mpi_reduce(&arg4, (double *)arg4.data);
    op_mpi_reduce(&arg5, (double *)arg5.data);
    op_profile_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_profile_exit();
}