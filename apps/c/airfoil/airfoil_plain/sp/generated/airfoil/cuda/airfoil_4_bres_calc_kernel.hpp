
namespace op2_k4 {
__device__ inline void bres_calc(const float *x1, const float *x2, const float *q1,
                      const float *adt1, float *res1, const int *bound) {
  float dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1_d * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));

  if (*bound == 1) {
    res1[1] += +p1 * dy;
    res1[2] += -p1 * dx;
  } else {
    vol1 = ri * (q1[1] * dy - q1[2] * dx);

    ri = 1.0f / qinf_d[0];
    p2 = gm1_d * (qinf_d[3] - 0.5f * ri * (qinf_d[1] * qinf_d[1] + qinf_d[2] * qinf_d[2]));
    vol2 = ri * (qinf_d[1] * dy - qinf_d[2] * dx);

    mu = (*adt1) * eps_d;

    f = 0.5f * (vol1 * q1[0] + vol2 * qinf_d[0]) + mu * (q1[0] - qinf_d[0]);
    res1[0] += f;
    f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * qinf_d[1] + p2 * dy) +
        mu * (q1[1] - qinf_d[1]);
    res1[1] += f;
    f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * qinf_d[2] - p2 * dx) +
        mu * (q1[2] - qinf_d[2]);
    res1[2] += f;
    f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (qinf_d[3] + p2)) +
        mu * (q1[3] - qinf_d[3]);
    res1[3] += f;
  }
}
}

__global__ void op_cuda_airfoil_4_bres_calc(
    const float *__restrict dat0,
    const float *__restrict dat1,
    const float *__restrict dat2,
    float *__restrict dat3,
    const int *__restrict dat4,
    const int *__restrict map0,
    const int *__restrict map1,
    
    int start,
    int end,
    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;

        float arg4_0_local[4];
        for (int d = 0; d < 4; ++d)
            arg4_0_local[d] = ZERO_float;

        op2_k4::bres_calc(
            dat0 + map0[round32(set_size) * 0 + n] * 2,
            dat0 + map0[round32(set_size) * 1 + n] * 2,
            dat1 + map1[round32(set_size) * 0 + n] * 4,
            dat2 + map1[round32(set_size) * 0 + n] * 1,
            arg4_0_local,
            dat4 + n * 1
        );

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat3 + map1[round32(set_size) * 0 + n] * 4 + d, arg4_0_local[d]);
    }
}

void op_par_loop_airfoil_4_bres_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    int num_args_expanded = 6;
    op_arg args_expanded[6];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(4);

    OP_kernels[4].name = name;
    OP_kernels[4].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): airfoil_4_bres_calc\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);



#ifdef OP_BLOCK_SIZE_4
    int block_size = OP_BLOCK_SIZE_4;
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

            op_cuda_airfoil_4_bres_calc<<<num_blocks, block_size>>>(
                (float *)arg0.data_d,
                (float *)arg2.data_d,
                (float *)arg3.data_d,
                (float *)arg4.data_d,
                (int *)arg5.data_d,
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
    OP_kernels[4].time += wall_end - wall_start;


}
