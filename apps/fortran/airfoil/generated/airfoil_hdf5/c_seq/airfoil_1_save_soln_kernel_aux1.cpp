#include "hydra_const_list_c_seq.h"

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

namespace op2_m_airfoil_1_save_soln_m {

static void save_soln(
    f2c::Ptr<const float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_qold
);


static void save_soln(
    f2c::Ptr<const float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_qold
) {
    const f2c::Span<const float, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> qold{_f2c_ptr_qold, f2c::Extent{1, 4}};
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

    op_timing2_enter_kernel("airfoil_1_save_soln", "c_seq", "Direct");

    op_timing2_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_timing2_next("Computation");



    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {


        op2_m_airfoil_1_save_soln_m::save_soln(
            (double *)arg0.data + n * 4,
            (double *)arg1.data + n * 4
        );

    }


    op_timing2_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_timing2_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_timing2_exit();
}