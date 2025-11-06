#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_5_update_m {

static void update(
    f2c::Ptr<const float> _f2c_ptr_qold,
    f2c::Ptr<float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_res,
    const float adt,
    f2c::Ptr<float> _f2c_ptr_rms,
    float& maxerr,
    const int idx,
    int& errloc
);


static void update(
    f2c::Ptr<const float> _f2c_ptr_qold,
    f2c::Ptr<float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_res,
    const float adt,
    f2c::Ptr<float> _f2c_ptr_rms,
    float& maxerr,
    const int idx,
    int& errloc
) {
    const f2c::Span<const float, 1> qold{_f2c_ptr_qold, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res{_f2c_ptr_res, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> rms{_f2c_ptr_rms, f2c::Extent{1, 2}};
    float del;
    float adti;
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

    op_timing2_enter_kernel("airfoil_5_update", "c_seq", "Direct");

    op_timing2_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_timing2_next("Computation");



    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

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


    op_timing2_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_timing2_next("MPI Reduce");

    op_mpi_reduce(&arg4, (double *)arg4.data);
    op_mpi_reduce(&arg5, (double *)arg5.data);
    op_timing2_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_timing2_exit();
}