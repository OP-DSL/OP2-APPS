namespace op2_k1 {
inline void res_calc(const double **x, const double **phim, double *K,
                     /*double *Kt,*/ double **res) {
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
      u[0] += N_x[j] * phim[j][0];
      u[1] += N_x[4 + j] * phim[j][0];
    }

    double Dk = 1.0 + 0.5 * gm1 * (m2 - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, gm1i); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    for (int j = 0; j < 4; j++) {
      res[j][0] += wt1 * rho * (u[0] * N_x[j] + u[1] * N_x[4 + j]);
    }
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        K[j * 4 + k] +=
            wt1 * rho * (N_x[j] * N_x[k] + N_x[4 + j] * N_x[4 + k]) -
            wt1 * rc2 * (u[0] * N_x[j] + u[1] * N_x[4 + j]) *
                (u[0] * N_x[k] + u[1] * N_x[4 + k]);
      }
    }
  }
}
}

void op_par_loop_aero_1_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    int num_args_expanded = 13;
    op_arg args_expanded[13];

    args_expanded[0] = op_arg_dat(arg0.dat, 0, arg0.map, 2, "double", 0);
    args_expanded[1] = op_arg_dat(arg0.dat, 1, arg0.map, 2, "double", 0);
    args_expanded[2] = op_arg_dat(arg0.dat, 2, arg0.map, 2, "double", 0);
    args_expanded[3] = op_arg_dat(arg0.dat, 3, arg0.map, 2, "double", 0);
    args_expanded[4] = op_arg_dat(arg1.dat, 0, arg1.map, 1, "double", 0);
    args_expanded[5] = op_arg_dat(arg1.dat, 1, arg1.map, 1, "double", 0);
    args_expanded[6] = op_arg_dat(arg1.dat, 2, arg1.map, 1, "double", 0);
    args_expanded[7] = op_arg_dat(arg1.dat, 3, arg1.map, 1, "double", 0);
    args_expanded[8] = arg2;
    args_expanded[9] = op_arg_dat(arg3.dat, 0, arg3.map, 1, "double", 3);
    args_expanded[10] = op_arg_dat(arg3.dat, 1, arg3.map, 1, "double", 3);
    args_expanded[11] = op_arg_dat(arg3.dat, 2, arg3.map, 1, "double", 3);
    args_expanded[12] = op_arg_dat(arg3.dat, 3, arg3.map, 1, "double", 3);

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_1_res_calc\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        if (n < set->core_size && n > 0 && n % OP_mpi_test_frequency == 0)
            op_mpi_test_all(num_args_expanded, args_expanded);

        if (n == set->core_size)
            op_mpi_wait_all(num_args_expanded, args_expanded);

        int *map0 = arg0.map_data + n * arg0.map->dim;

        const double *arg0_vec[] = {
            (double *)arg0.data + map0[0] * 2,
            (double *)arg0.data + map0[1] * 2,
            (double *)arg0.data + map0[2] * 2,
            (double *)arg0.data + map0[3] * 2
        };

        const double *arg1_vec[] = {
            (double *)arg1.data + map0[0] * 1,
            (double *)arg1.data + map0[1] * 1,
            (double *)arg1.data + map0[2] * 1,
            (double *)arg1.data + map0[3] * 1
        };

        double *arg3_vec[] = {
            (double *)arg3.data + map0[0] * 1,
            (double *)arg3.data + map0[1] * 1,
            (double *)arg3.data + map0[2] * 1,
            (double *)arg3.data + map0[3] * 1
        };

        op2_k1::res_calc(
            arg0_vec,
            arg1_vec,
            (double *)arg2.data + n * 16,
            arg3_vec
        );
    }

    if (set_size == 0 || set_size == set->core_size)
        op_mpi_wait_all(num_args_expanded, args_expanded);

    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;

}
