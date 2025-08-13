#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_airfoil_mpi_4_bres_calc {

inline void bres_calc(const float *x1, const float *x2, const float *q1,
                      const float *adt1, float *res1, const int *bound) {
  float dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1 * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));

  if (*bound == 1) {
    res1[1] += +p1 * dy;
    res1[2] += -p1 * dx;
  } else {
    vol1 = ri * (q1[1] * dy - q1[2] * dx);

    ri = 1.0f / qinf[0];
    p2 = gm1 * (qinf[3] - 0.5f * ri * (qinf[1] * qinf[1] + qinf[2] * qinf[2]));
    vol2 = ri * (qinf[1] * dy - qinf[2] * dx);

    mu = (*adt1) * eps;

    f = 0.5f * (vol1 * q1[0] + vol2 * qinf[0]) + mu * (q1[0] - qinf[0]);
    res1[0] += f;
    f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * qinf[1] + p2 * dy) +
        mu * (q1[1] - qinf[1]);
    res1[1] += f;
    f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * qinf[2] - p2 * dx) +
        mu * (q1[2] - qinf[2]);
    res1[2] += f;
    f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (qinf[3] + p2)) +
        mu * (q1[3] - qinf[3]);
    res1[3] += f;
  }
}}


void op_par_loop_airfoil_mpi_4_bres_calc(
    const char* name,
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

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;
        int *map1 = arg2.map_data + n * arg2.map->dim;


        op2_m_airfoil_mpi_4_bres_calc::bres_calc(
            (float *)arg0.data + map0[0] * 2,
            (float *)arg1.data + map0[1] * 2,
            (float *)arg2.data + map1[0] * 4,
            (float *)arg3.data + map1[0] * 1,
            (float *)arg4.data + map1[0] * 4,
            (int *)arg5.data + n * 1
        );

        if (n == set->size - 1) {
        }
    }

    if (n_exec < set->size) {
    }

    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);


    op_mpi_set_dirtybit(n_args, args);
}