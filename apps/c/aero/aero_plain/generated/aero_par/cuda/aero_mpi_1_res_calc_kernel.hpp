namespace op2_k1 {


__device__ inline void res_calc(const double *x0, const double *x1, const double *x2, const double *x3,
                     const double *phim0, const double *phim1, const double *phim2, const double *phim3,
                     double *K, /*double *Kt,*/ double *res0, double *res1, double *res2, double *res3) {
  double x[4][2], phim[4];
  x[0][0] = x0[0]; x[1][0] = x1[0]; x[2][0] = x2[0]; x[3][0] = x3[0];
  x[0][1] = x0[1]; x[1][1] = x1[1]; x[2][1] = x2[1]; x[3][1] = x3[1];
  phim[0] = phim0[0]; phim[1] = phim1[0]; phim[2] = phim2[0]; phim[3] = phim3[0];

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
      u[0] += N_x[j] * phim[j];
      u[1] += N_x[4 + j] * phim[j];
    }

    double Dk = 1.0 + 0.5 * gm1_d * (m2_d - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, gm1i_d); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    res0[0] += wt1 * rho * (u[0] * N_x[0] + u[1] * N_x[4 + 0]);
    res1[0] += wt1 * rho * (u[0] * N_x[1] + u[1] * N_x[4 + 1]);
    res2[0] += wt1 * rho * (u[0] * N_x[2] + u[1] * N_x[4 + 2]);
    res3[0] += wt1 * rho * (u[0] * N_x[3] + u[1] * N_x[4 + 3]);

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
        double arg9_0_local[1];
        for (int d = 0; d < 1; ++d)
            arg9_0_local[d] = ZERO_double;

        double arg10_1_local[1];
        for (int d = 0; d < 1; ++d)
            arg10_1_local[d] = ZERO_double;

        double arg11_2_local[1];
        for (int d = 0; d < 1; ++d)
            arg11_2_local[d] = ZERO_double;

        double arg12_3_local[1];
        for (int d = 0; d < 1; ++d)
            arg12_3_local[d] = ZERO_double;

        op2_k1::res_calc(
            dat0 + map0[round32(set_size) * 0 + n] * 2,
            dat0 + map0[round32(set_size) * 1 + n] * 2,
            dat0 + map0[round32(set_size) * 2 + n] * 2,
            dat0 + map0[round32(set_size) * 3 + n] * 2,
            dat1 + map0[round32(set_size) * 0 + n] * 1,
            dat1 + map0[round32(set_size) * 1 + n] * 1,
            dat1 + map0[round32(set_size) * 2 + n] * 1,
            dat1 + map0[round32(set_size) * 3 + n] * 1,
            dat2 + n * 16,
            arg9_0_local,
            arg10_1_local,
            arg11_2_local,
            arg12_3_local
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 0 + n] * 1 + d, arg9_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 1 + n] * 1 + d, arg10_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 2 + n] * 1 + d, arg11_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[round32(set_size) * 3 + n] * 1 + d, arg12_3_local[d]);
    }
}

void op_par_loop_aero_mpi_1_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7,
    op_arg arg8,
    op_arg arg9,
    op_arg arg10,
    op_arg arg11,
    op_arg arg12
) {
    int num_args_expanded = 13;
    op_arg args_expanded[13];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;
    args_expanded[6] = arg6;
    args_expanded[7] = arg7;
    args_expanded[8] = arg8;
    args_expanded[9] = arg9;
    args_expanded[10] = arg10;
    args_expanded[11] = arg11;
    args_expanded[12] = arg12;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    op_profile_enter_kernel(name, "", "Indirect");
    op_profile_enter("MPI Exchanges");

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_mpi_1_res_calc\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);

    op_profile_next("Computation");



#ifdef OP_BLOCK_SIZE_1
    int block_size = OP_BLOCK_SIZE_1;
#else
    int block_size = OP_block_size;
#endif

    for (int round = 0; round < 2; ++round ) {
        if (round == 1) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all_grouped(num_args_expanded, args_expanded, 2);
            op_profile_next("Computation");
        }

        int start = round == 0 ? 0 : set->core_size;
        int end = round == 0 ? set->core_size : set->size + set->exec_size;

        if (end - start > 0) {
            int num_blocks = (end - start - 1) / block_size + 1;

            op_cuda_aero_mpi_1_res_calc<<<num_blocks, block_size>>>(
                (double *)arg0.data_d,
                (double *)arg4.data_d,
                (double *)arg8.data_d,
                (double *)arg9.data_d,
                arg0.map_data_d,
                start,
                end,
                set->size + set->exec_size
            );
        }
    }

    op_profile_exit();

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());
    op_profile_exit();

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;


}
