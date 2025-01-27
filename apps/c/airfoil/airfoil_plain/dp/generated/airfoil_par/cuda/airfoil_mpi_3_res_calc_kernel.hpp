
namespace op2_k3 {
__device__ inline void res_calc(const double *x1, const double *x2, const double *q1,
                     const double *q2, const double *adt1, const double *adt2,
                     double *res1, double *res2) {
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1_d * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));
  vol1 = ri * (q1[1] * dy - q1[2] * dx);

  ri = 1.0f / q2[0];
  p2 = gm1_d * (q2[3] - 0.5f * ri * (q2[1] * q2[1] + q2[2] * q2[2]));
  vol2 = ri * (q2[1] * dy - q2[2] * dx);

  mu = 0.5f * ((*adt1) + (*adt2)) * eps_d;

  f = 0.5f * (vol1 * q1[0] + vol2 * q2[0]) + mu * (q1[0] - q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * q2[1] + p2 * dy) +
      mu * (q1[1] - q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * q2[2] - p2 * dx) +
      mu * (q1[2] - q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (q2[3] + p2)) + mu * (q1[3] - q2[3]);
  res1[3] += f;
  res2[3] -= f;
}
}

__global__ void op_cuda_airfoil_mpi_3_res_calc(
    const double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict map0,
    const int *__restrict map1,
    
    int start,
    int end,
    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;

        double arg6_0_local[4];
        for (int d = 0; d < 4; ++d)
            arg6_0_local[d] = ZERO_double;

        double arg7_1_local[4];
        for (int d = 0; d < 4; ++d)
            arg7_1_local[d] = ZERO_double;

        op2_k3::res_calc(
            dat0 + map0[round32(set_size) * 0 + n] * 2,
            dat0 + map0[round32(set_size) * 1 + n] * 2,
            dat1 + map1[round32(set_size) * 0 + n] * 4,
            dat1 + map1[round32(set_size) * 1 + n] * 4,
            dat2 + map1[round32(set_size) * 0 + n] * 1,
            dat2 + map1[round32(set_size) * 1 + n] * 1,
            arg6_0_local,
            arg7_1_local
        );

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat3 + map1[round32(set_size) * 0 + n] * 4 + d, arg6_0_local[d]);

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat3 + map1[round32(set_size) * 1 + n] * 4 + d, arg7_1_local[d]);
    }
}

void op_par_loop_airfoil_mpi_3_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7
) {
    int num_args_expanded = 8;
    op_arg args_expanded[8];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;
    args_expanded[6] = arg6;
    args_expanded[7] = arg7;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(3);

    OP_kernels[3].name = name;
    OP_kernels[3].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): airfoil_mpi_3_res_calc\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);



#ifdef OP_BLOCK_SIZE_3
    int block_size = OP_BLOCK_SIZE_3;
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

            op_cuda_airfoil_mpi_3_res_calc<<<num_blocks, block_size>>>(
                (double *)arg0.data_d,
                (double *)arg2.data_d,
                (double *)arg4.data_d,
                (double *)arg6.data_d,
                arg0.map_data_d,
                arg2.map_data_d,
                start,
                end,
                set->size + set->exec_size
            );
        }
    }

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[3].time += wall_end - wall_start;


}
