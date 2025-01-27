namespace op2_k1 {
inline void res(const double *A, const double *u, double *du,
                const double *beta, const int *index, const int *idx_ppedge0,
                const int *idx_ppedge1) {
  *du += (*beta) * (*A) * (*u);
  printf("edge %d, nodes %d, %d\n", *index, *idx_ppedge0, *idx_ppedge1);
}
}

void op_par_loop_jac_1_res(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6
) {
    int num_args_expanded = 4;
    op_arg args_expanded[4];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): jac_1_res\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        if (n < set->core_size && n > 0 && n % OP_mpi_test_frequency == 0)
            op_mpi_test_all(num_args_expanded, args_expanded);

        if (n == set->core_size)
            op_mpi_wait_all(num_args_expanded, args_expanded);

        int *map0 = arg1.map_data + n * arg1.map->dim;

        op2_k1::res(
            (double *)arg0.data + n * 1,
            (double *)arg1.data + map0[1] * 1,
            (double *)arg2.data + map0[0] * 1,
            (double *)arg3.data,
            &n,
            &map0[0],
            &map0[1]
        );
    }

    if (set_size == 0 || set_size == set->core_size)
        op_mpi_wait_all(num_args_expanded, args_expanded);

    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;

}
