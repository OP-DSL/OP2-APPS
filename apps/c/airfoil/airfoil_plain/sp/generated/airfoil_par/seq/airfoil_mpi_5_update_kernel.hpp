#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_airfoil_mpi_5_update {

inline void update(const float *qold, float *q, float *res, const float *adt,
                   float *rms) {
  float del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    *rms += del * del;
  }
}}


void op_par_loop_airfoil_mpi_5_update(
    const char* name,
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

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {


        op2_m_airfoil_mpi_5_update::update(
            (float *)arg0.data + n * 4,
            (float *)arg1.data + n * 4,
            (float *)arg2.data + n * 4,
            (float *)arg3.data + n * 1,
            (float *)arg4.data
        );

    }


    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_mpi_reduce(&arg4, (float *)arg4.data);

    op_mpi_set_dirtybit(n_args, args);
}