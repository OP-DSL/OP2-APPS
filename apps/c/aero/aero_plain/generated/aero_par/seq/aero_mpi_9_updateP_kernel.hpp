namespace op2_k9 {
inline void updateP(const double *r, double *p, const double *beta) {
  *p = (*beta) * (*p) + (*r);
}
}

void op_par_loop_aero_mpi_9_updateP(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int num_args_expanded = 3;
    op_arg args_expanded[3];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(9);

    OP_kernels[9].name = name;
    OP_kernels[9].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_9_updateP\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k9::updateP(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + n * 1,
            (double *)arg2.data
        );
    }

    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[9].time += wall_end - wall_start;

}
