#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_aero_1_res_calc {

inline void res_calc(const double *x0, const double *x1, const double *x2, const double *x3,
                     const double *phim0, const double *phim1, const double *phim2, const double *phim3,
                     double *K, /*double *Kt,*/ double *res0, double *res1, double *res2, double *res3) {
  double x[4][2], phim[4];
  x[0][0] = x0[0]; x[1][0] = x1[0]; x[2][0] = x2[0]; x[3][0] = x3[0];
  x[0][1] = x0[1]; x[1][1] = x1[1]; x[2][1] = x2[1]; x[3][1] = x3[1];
  phim[0] = phim0[0]; phim[1] = phim1[0]; phim[2] = phim2[0]; phim[3] = phim3[0];

  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      K[j * 4 + k] = 0;
    }
  }
  for (int i = 0; i < 4; i++) { // for each gauss point
    double det_x_xi = 0;
    double N_x[8];

    double a = 0;
    for (int m = 0; m < 4; m++)
      det_x_xi += Ng2_xi[4 * i + 16 + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] = det_x_xi * Ng2_xi[4 * i + m];

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi[4 * i + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] = a * Ng2_xi[4 * i + 16 + m];

    det_x_xi *= a;

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi[4 * i + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] -= a * Ng2_xi[4 * i + 16 + m];

    double b = 0;
    for (int m = 0; m < 4; m++)
      b += Ng2_xi[4 * i + 16 + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] -= b * Ng2_xi[4 * i + m];

    det_x_xi -= a * b;

    for (int j = 0; j < 8; j++)
      N_x[j] /= det_x_xi;

    double wt1 = wtg2[i] * det_x_xi;
    // double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0; j < 4; j++) {
      u[0] += N_x[j] * phim[j];
      u[1] += N_x[4 + j] * phim[j];
    }

    double Dk = 1.0 + 0.5 * gm1 * (m2 - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, gm1i); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    res0[0] += wt1 * rho * (u[0] * N_x[0] + u[1] * N_x[4 + 0]);
    res1[0] += wt1 * rho * (u[0] * N_x[1] + u[1] * N_x[4 + 1]);
    res2[0] += wt1 * rho * (u[0] * N_x[2] + u[1] * N_x[4 + 2]);
    res3[0] += wt1 * rho * (u[0] * N_x[3] + u[1] * N_x[4 + 3]);

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        K[j * 4 + k] +=
            wt1 * rho * (N_x[j] * N_x[k] + N_x[4 + j] * N_x[4 + k]) -
            wt1 * rc2 * (u[0] * N_x[j] + u[1] * N_x[4 + j]) *
                (u[0] * N_x[k] + u[1] * N_x[4 + k]);
      }
    }
  }
}}


void op_par_loop_aero_1_res_calc(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7,
    op_arg arg8,
    op_arg arg9,
    op_arg arg10,
    op_arg arg11,
    op_arg arg12
) {
    int n_args = 13;
    op_arg args[13];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;
    args[11] = arg11;
    args[12] = arg12;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_aero_1_res_calc::res_calc(
            (double *)arg0.data + map0[0] * 2,
            (double *)arg1.data + map0[1] * 2,
            (double *)arg2.data + map0[2] * 2,
            (double *)arg3.data + map0[3] * 2,
            (double *)arg4.data + map0[0] * 1,
            (double *)arg5.data + map0[1] * 1,
            (double *)arg6.data + map0[2] * 1,
            (double *)arg7.data + map0[3] * 1,
            (double *)arg8.data + n * 16,
            (double *)arg9.data + map0[0] * 1,
            (double *)arg10.data + map0[1] * 1,
            (double *)arg11.data + map0[2] * 1,
            (double *)arg12.data + map0[3] * 1
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