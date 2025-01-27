namespace op2_k5 {
inline void update(const float *qold, float *q, float *res, const float *adt,
                   float *rms) {
  float del, adti;
  float rmsl = 0.0f;
  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    rmsl += del * del;
  }
  *rms += rmsl;
}
}

void op_par_loop_airfoil_5_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    int num_args_expanded = 5;
    op_arg args_expanded[5];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(5);

    OP_kernels[5].name = name;
    OP_kernels[5].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): airfoil_5_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k5::update(
            (float *)arg0.data + n * 4,
            (float *)arg1.data + n * 4,
            (float *)arg2.data + n * 4,
            (float *)arg3.data + n * 1,
            (float *)arg4.data
        );
    }

    op_mpi_reduce(&arg4, (float *)arg4.data);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[5].time += wall_end - wall_start;

}
