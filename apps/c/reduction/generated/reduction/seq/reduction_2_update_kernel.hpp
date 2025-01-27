namespace op2_k2 {
inline void update(double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}
}

void op_par_loop_reduction_2_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    int num_args_expanded = 2;
    op_arg args_expanded[2];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): reduction_2_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k2::update(
            (double *)arg0.data + n * 4,
            (int *)arg1.data
        );
    }

    op_mpi_reduce(&arg1, (int *)arg1.data);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[2].time += wall_end - wall_start;

}
