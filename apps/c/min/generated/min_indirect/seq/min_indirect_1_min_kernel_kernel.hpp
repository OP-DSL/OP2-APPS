namespace op2_k1 {
void min_kernel(const int *d, int *min) {
    *min = std::min(*d, *min);
}
}

void op_par_loop_min_indirect_1_min_kernel(
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
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): min_indirect_1_min_kernel\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int arg1_local[1] = {0};

    memcpy(arg1_local, arg1.data, 1 * sizeof(int));


    for (int n = 0; n < set_size; ++n) {
        if (n < set->core_size && n > 0 && n % OP_mpi_test_frequency == 0)
            op_mpi_test_all(num_args_expanded, args_expanded);

        if (n == set->core_size)
            op_mpi_wait_all(num_args_expanded, args_expanded);

        int *map0 = arg0.map_data + n * arg0.map->dim;

        if (n == set->size) {
            memcpy(arg1.data, arg1_local, 1 * sizeof(int));
        }

        op2_k1::min_kernel(
            (int *)arg0.data + map0[0] * 1,
            arg1_local
        );
    }

    if (set_size == 0 || set_size == set->core_size)
        op_mpi_wait_all(num_args_expanded, args_expanded);

    op_mpi_reduce(&arg1, (int *)arg1.data);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;

}
