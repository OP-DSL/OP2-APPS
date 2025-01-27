
namespace op2_k1 {
__device__ inline void res_calc(const double **x, const double **phim, double *K,
                     /*double *Kt,*/ double **res) {
  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      K[j * 4 + k] = 0;
    }
  }
  for (int i = 0; i < 4; i++) { // for each gauss point
    double det_x_xi = 0;
    double N_x[8];

    double a = 0;
    for (int m = 0; m < 4; m++)
      det_x_xi += Ng2_xi_d[4 * i + 16 + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] = det_x_xi * Ng2_xi_d[4 * i + m];

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi_d[4 * i + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] = a * Ng2_xi_d[4 * i + 16 + m];

    det_x_xi *= a;

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi_d[4 * i + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] -= a * Ng2_xi_d[4 * i + 16 + m];

    double b = 0;
    for (int m = 0; m < 4; m++)
      b += Ng2_xi_d[4 * i + 16 + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] -= b * Ng2_xi_d[4 * i + m];

    det_x_xi -= a * b;

    for (int j = 0; j < 8; j++)
      N_x[j] /= det_x_xi;

    double wt1 = wtg2_d[i] * det_x_xi;
    // double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0; j < 4; j++) {
      u[0] += N_x[j] * phim[j][0];
      u[1] += N_x[4 + j] * phim[j][0];
    }

    double Dk = 1.0 + 0.5 * gm1_d * (m2_d - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, gm1i_d); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    for (int j = 0; j < 4; j++) {
      res[j][0] += wt1 * rho * (u[0] * N_x[j] + u[1] * N_x[4 + j]);
    }
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        K[j * 4 + k] +=
            wt1 * rho * (N_x[j] * N_x[k] + N_x[4 + j] * N_x[4 + k]) -
            wt1 * rc2 * (u[0] * N_x[j] + u[1] * N_x[4 + j]) *
                (u[0] * N_x[k] + u[1] * N_x[4 + k]);
      }
    }
  }
}
}

__global__ void op_cuda_aero_mpi_1_res_calc(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict map0,
    
    int start,
    int end,
    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;

        double arg3_0_local[1];
        for (int d = 0; d < 1; ++d)
            arg3_0_local[d] = ZERO_double;

        double arg3_1_local[1];
        for (int d = 0; d < 1; ++d)
            arg3_1_local[d] = ZERO_double;

        double arg3_2_local[1];
        for (int d = 0; d < 1; ++d)
            arg3_2_local[d] = ZERO_double;

        double arg3_3_local[1];
        for (int d = 0; d < 1; ++d)
            arg3_3_local[d] = ZERO_double;

        const double *arg0_vec[4];
        arg0_vec[0] = dat0 + map0[round32(set_size) * 0 + n] * 2;
        arg0_vec[1] = dat0 + map0[round32(set_size) * 1 + n] * 2;
        arg0_vec[2] = dat0 + map0[round32(set_size) * 2 + n] * 2;
        arg0_vec[3] = dat0 + map0[round32(set_size) * 3 + n] * 2;

        const double *arg1_vec[4];
        arg1_vec[0] = dat1 + map0[round32(set_size) * 0 + n] * 1;
        arg1_vec[1] = dat1 + map0[round32(set_size) * 1 + n] * 1;
        arg1_vec[2] = dat1 + map0[round32(set_size) * 2 + n] * 1;
        arg1_vec[3] = dat1 + map0[round32(set_size) * 3 + n] * 1;

        double *arg3_vec[4];
        arg3_vec[0] = arg3_0_local;
        arg3_vec[1] = arg3_1_local;
        arg3_vec[2] = arg3_2_local;
        arg3_vec[3] = arg3_3_local;

        op2_k1::res_calc(
            arg0_vec,
            arg1_vec,
            dat2 + n * 16,
            arg3_vec
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 0 + n] * 1 + d, arg3_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 1 + n] * 1 + d, arg3_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 2 + n] * 1 + d, arg3_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 3 + n] * 1 + d, arg3_3_local[d]);
    }
}

void op_par_loop_aero_mpi_1_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    int num_args_expanded = 13;
    op_arg args_expanded[13];

    args_expanded[0] = op_arg_dat(arg0.dat, 0, arg0.map, 2, "double", 0);
    args_expanded[1] = op_arg_dat(arg0.dat, 1, arg0.map, 2, "double", 0);
    args_expanded[2] = op_arg_dat(arg0.dat, 2, arg0.map, 2, "double", 0);
    args_expanded[3] = op_arg_dat(arg0.dat, 3, arg0.map, 2, "double", 0);
    args_expanded[4] = op_arg_dat(arg1.dat, 0, arg1.map, 1, "double", 0);
    args_expanded[5] = op_arg_dat(arg1.dat, 1, arg1.map, 1, "double", 0);
    args_expanded[6] = op_arg_dat(arg1.dat, 2, arg1.map, 1, "double", 0);
    args_expanded[7] = op_arg_dat(arg1.dat, 3, arg1.map, 1, "double", 0);
    args_expanded[8] = arg2;
    args_expanded[9] = op_arg_dat(arg3.dat, 0, arg3.map, 1, "double", 3);
    args_expanded[10] = op_arg_dat(arg3.dat, 1, arg3.map, 1, "double", 3);
    args_expanded[11] = op_arg_dat(arg3.dat, 2, arg3.map, 1, "double", 3);
    args_expanded[12] = op_arg_dat(arg3.dat, 3, arg3.map, 1, "double", 3);

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_mpi_1_res_calc\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);



#ifdef OP_BLOCK_SIZE_1
    int block_size = OP_BLOCK_SIZE_1;
#else
    int block_size = OP_block_size;
#endif

    for (int round = 0; round < 2; ++round ) {
        if (round == 1)
            op_mpi_wait_all_grouped(num_args_expanded, args_expanded, 2);

        int start = round == 0 ? 0 : set->core_size;
        int end = round == 0 ? set->core_size : set->size + set->exec_size;

        if (end - start > 0) {
            int num_blocks = (end - start - 1) / block_size + 1;

            op_cuda_aero_mpi_1_res_calc<<<num_blocks, block_size>>>(
                (double *)arg0.data_d,
                (double *)arg1.data_d,
                (double *)arg2.data_d,
                (double *)arg3.data_d,
                arg0.map_data_d,
                start,
                end,
                set->size + set->exec_size
            );
        }
    }

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;


}
