namespace op2_m_jac1_mpi_2_update_kernel_m {

static void update_kernel(
    const double r,
    double& du,
    double& u,
    double& u_sum,
    double& u_max
);


static void update_kernel(
    const double r,
    double& du,
    double& u,
    double& u_sum,
    double& u_max
) {

    u = u + du + op2_const_alpha * r;
    du = 0.0;
    u_sum = u_sum + f2c::pow(u, 2);
    u_max = f2c::max(u_max, u);
}

}


extern "C" void op2_k_jac1_mpi_2_update_kernel_m_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    int n_args = 5;
    op_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    op_profile_enter_kernel("jac1_mpi_2_update_kernel", "c_seq", "Direct");

    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, n_args, args);

    op_profile_next("Computation");



    [[maybe_unused]] int zero_int = 0;
    [[maybe_unused]] bool zero_bool = 0;
    [[maybe_unused]] float zero_float = 0;
    [[maybe_unused]] double zero_double = 0;

    for (int n = 0; n < n_exec; ++n) {


        op2_m_jac1_mpi_2_update_kernel_m::update_kernel(
            ((double *)arg0.data + n * 1)[0],
            ((double *)arg1.data + n * 1)[0],
            ((double *)arg2.data + n * 1)[0],
            ((double *)arg3.data)[0],
            ((double *)arg4.data)[0]
        );

    }


    op_profile_next("MPI Wait");
    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);

    op_profile_next("MPI Reduce");

    op_mpi_reduce(&arg3, (double *)arg3.data);
    op_mpi_reduce(&arg4, (double *)arg4.data);
    op_profile_exit();

    op_mpi_set_dirtybit(n_args, args);
    op_profile_exit();
}