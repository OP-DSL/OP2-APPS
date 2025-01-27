namespace op2_k7 {
inline void updateUR(double *u, double *r, const double *p, double *v,
                     const double *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}
}

void op_par_loop_aero_mpi_7_updateUR(
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
    op_timing_realloc(7);

    OP_kernels[7].name = name;
    OP_kernels[7].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_7_updateUR\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k7::updateUR(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + n * 1,
            (double *)arg2.data + n * 1,
            (double *)arg3.data + n * 1,
            (double *)arg4.data
        );
    }

    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[7].time += wall_end - wall_start;

}
