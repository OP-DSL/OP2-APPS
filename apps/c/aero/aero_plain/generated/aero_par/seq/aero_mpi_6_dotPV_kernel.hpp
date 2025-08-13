#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_aero_mpi_6_dotPV {

inline void dotPV(const double *p, const double *v, double *c) { *c += (*p) * (*v); }}


void op_par_loop_aero_mpi_6_dotPV(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int n_args = 3;
    op_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {


        op2_m_aero_mpi_6_dotPV::dotPV(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + n * 1,
            (double *)arg2.data
        );

    }


    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_mpi_reduce(&arg2, (double *)arg2.data);

    op_mpi_set_dirtybit(n_args, args);
}