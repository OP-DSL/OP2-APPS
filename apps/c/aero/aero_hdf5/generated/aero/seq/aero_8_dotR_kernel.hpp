namespace op2_k8 {
inline void dotR(const double *r, double *c) { *c += (*r) * (*r); }
}

void op_par_loop_aero_8_dotR(
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
    op_timing_realloc(8);

    OP_kernels[8].name = name;
    OP_kernels[8].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_8_dotR\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k8::dotR(
            (double *)arg0.data + n * 1,
            (double *)arg1.data
        );
    }

    op_mpi_reduce(&arg1, (double *)arg1.data);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[8].time += wall_end - wall_start;

}
