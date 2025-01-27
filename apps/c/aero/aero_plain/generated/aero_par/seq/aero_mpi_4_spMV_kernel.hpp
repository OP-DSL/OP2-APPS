namespace op2_k4 {
inline void spMV(double **v, const double *K, const double **p) {
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += OP2_STRIDE(K, (j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += OP2_STRIDE(K, (j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = OP2_STRIDE(K, (j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v[0][0] += K[0] * p[0][0];
  v[0][0] += K[1] * p[1][0];
  v[1][0] += K[1] * p[0][0];
  v[0][0] += K[2] * p[2][0];
  v[2][0] += K[2] * p[0][0];
  v[0][0] += K[3] * p[3][0];
  v[3][0] += K[3] * p[0][0];
  v[1][0] += K[4 + 1] * p[1][0];
  v[1][0] += K[4 + 2] * p[2][0];
  v[2][0] += K[4 + 2] * p[1][0];
  v[1][0] += K[4 + 3] * p[3][0];
  v[3][0] += K[4 + 3] * p[1][0];
  v[2][0] += K[8 + 2] * p[2][0];
  v[2][0] += K[8 + 3] * p[3][0];
  v[3][0] += K[8 + 3] * p[2][0];
  v[3][0] += K[15] * p[3][0];
}
}

void op_par_loop_aero_mpi_4_spMV(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int num_args_expanded = 9;
    op_arg args_expanded[9];

    args_expanded[0] = op_arg_dat(arg0.dat, 0, arg0.map, 1, "double", 3);
    args_expanded[1] = op_arg_dat(arg0.dat, 1, arg0.map, 1, "double", 3);
    args_expanded[2] = op_arg_dat(arg0.dat, 2, arg0.map, 1, "double", 3);
    args_expanded[3] = op_arg_dat(arg0.dat, 3, arg0.map, 1, "double", 3);
    args_expanded[4] = arg1;
    args_expanded[5] = op_arg_dat(arg2.dat, 0, arg2.map, 1, "double", 0);
    args_expanded[6] = op_arg_dat(arg2.dat, 1, arg2.map, 1, "double", 0);
    args_expanded[7] = op_arg_dat(arg2.dat, 2, arg2.map, 1, "double", 0);
    args_expanded[8] = op_arg_dat(arg2.dat, 3, arg2.map, 1, "double", 0);

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(4);

    OP_kernels[4].name = name;
    OP_kernels[4].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_mpi_4_spMV\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        if (n < set->core_size && n > 0 && n % OP_mpi_test_frequency == 0)
            op_mpi_test_all(num_args_expanded, args_expanded);

        if (n == set->core_size)
            op_mpi_wait_all(num_args_expanded, args_expanded);

        int *map0 = arg0.map_data + n * arg0.map->dim;

        double *arg0_vec[] = {
            (double *)arg0.data + map0[0] * 1,
            (double *)arg0.data + map0[1] * 1,
            (double *)arg0.data + map0[2] * 1,
            (double *)arg0.data + map0[3] * 1
        };

        const double *arg2_vec[] = {
            (double *)arg2.data + map0[0] * 1,
            (double *)arg2.data + map0[1] * 1,
            (double *)arg2.data + map0[2] * 1,
            (double *)arg2.data + map0[3] * 1
        };

        op2_k4::spMV(
            arg0_vec,
            (double *)arg1.data + n * 16,
            arg2_vec
        );
    }

    if (set_size == 0 || set_size == set->core_size)
        op_mpi_wait_all(num_args_expanded, args_expanded);

    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[4].time += wall_end - wall_start;

}
