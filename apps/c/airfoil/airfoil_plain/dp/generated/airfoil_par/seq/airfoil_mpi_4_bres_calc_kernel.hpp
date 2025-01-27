namespace op2_k4 {
inline void bres_calc(const double *x1, const double *x2, const double *q1,
                      const double *adt1, double *res1, const int *bound) {
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

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
}
}

void op_par_loop_airfoil_mpi_4_bres_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    int num_args_expanded = 6;
    op_arg args_expanded[6];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(4);

    OP_kernels[4].name = name;
    OP_kernels[4].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): airfoil_mpi_4_bres_calc\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        if (n < set->core_size && n > 0 && n % OP_mpi_test_frequency == 0)
            op_mpi_test_all(num_args_expanded, args_expanded);

        if (n == set->core_size)
            op_mpi_wait_all(num_args_expanded, args_expanded);

        int *map0 = arg0.map_data + n * arg0.map->dim;
        int *map1 = arg2.map_data + n * arg2.map->dim;

        op2_k4::bres_calc(
            (double *)arg0.data + map0[0] * 2,
            (double *)arg1.data + map0[1] * 2,
            (double *)arg2.data + map1[0] * 4,
            (double *)arg3.data + map1[0] * 1,
            (double *)arg4.data + map1[0] * 4,
            (int *)arg5.data + n * 1
        );
    }

    if (set_size == 0 || set_size == set->core_size)
        op_mpi_wait_all(num_args_expanded, args_expanded);

    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[4].time += wall_end - wall_start;

}
