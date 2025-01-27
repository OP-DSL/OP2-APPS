namespace op2_k10 {
inline void update(double *phim, double *res, const double *u, double *rms) {
  *phim -= *u;
  *res = 0.0;
  *rms += (*u) * (*u);
}
}

void op_par_loop_aero_mpi_10_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    int num_args_expanded = 4;
    op_arg args_expanded[4];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(10);

    OP_kernels[10].name = name;
    OP_kernels[10].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_10_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k10::update(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + n * 1,
            (double *)arg2.data + n * 1,
            (double *)arg3.data
        );
    }

    op_mpi_reduce(&arg3, (double *)arg3.data);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[10].time += wall_end - wall_start;

}
