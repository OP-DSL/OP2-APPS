namespace op2_k6 {
inline void dotPV(const double *p, const double *v, double *c) { *c += (*p) * (*v); }
}

void op_par_loop_aero_mpi_6_dotPV(
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
    op_timing_realloc(6);

    OP_kernels[6].name = name;
    OP_kernels[6].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_6_dotPV\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k6::dotPV(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + n * 1,
            (double *)arg2.data
        );
    }

    op_mpi_reduce(&arg2, (double *)arg2.data);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[6].time += wall_end - wall_start;

}
