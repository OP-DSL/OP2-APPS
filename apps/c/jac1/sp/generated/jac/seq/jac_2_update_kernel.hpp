namespace op2_k2 {
inline void update(const float *r, float *du, float *u, float *u_sum,
                   float *u_max) {
  *u += *du + alpha * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = ((*u_max > *u) ? (*u_max) : (*u));
}
}

void op_par_loop_jac_2_update(
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
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): jac_2_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


    for (int n = 0; n < set_size; ++n) {
        op2_k2::update(
            (float *)arg0.data + n * 1,
            (float *)arg1.data + n * 1,
            (float *)arg2.data + n * 1,
            (float *)arg3.data,
            (float *)arg4.data
        );
    }

    op_mpi_reduce(&arg3, (float *)arg3.data);
    op_mpi_reduce(&arg4, (float *)arg4.data);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[2].time += wall_end - wall_start;

}
