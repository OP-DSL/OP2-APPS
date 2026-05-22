#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_airfoil_1_save_soln {

inline void save_soln(const double *q, double *qold) {
  for (int n = 0; n < 4; n++)
    qold[n] = q[n];
}}


void op_par_loop_airfoil_1_save_soln(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    int n_args = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    op_profile_enter_kernel("airfoil_1_save_soln", "seq", "Direct");

    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_profile_next("Computation");



    for (int n = 0; n < n_exec; ++n) {


        op2_m_airfoil_1_save_soln::save_soln(
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