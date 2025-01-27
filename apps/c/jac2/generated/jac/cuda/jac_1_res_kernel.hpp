
namespace op2_k1 {
__device__ inline void res(const double *A, const float *u, float *du, const float *beta) {
  *du += (float)((*beta) * (*A) * (*u));
}
}

__global__ void op_cuda_jac_1_res(
    const double *__restrict dat0,
    const float *__restrict dat1,
    float *__restrict dat2,
    const int *__restrict map0,
    float *gbl3,
    
    int start,
    int end,
    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;

        float arg2_0_local[3];
        for (int d = 0; d < 3; ++d)
            arg2_0_local[d] = ZERO_float;

        op2_k1::res(
            dat0 + n * 3,
            dat1 + map0[round32(set_size) * 1 + n] * 2,
            arg2_0_local,
            gbl3
        );

        for (int d = 0; d < 3; ++d)
            atomicAdd(dat2 + map0[round32(set_size) * 0 + n] * 3 + d, arg2_0_local[d]);
    }
}

void op_par_loop_jac_1_res(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
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

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    float *arg3_host_data = (float *)arg3.data;

    int const_bytes = 0;

    const_bytes += ROUND_UP(1 * sizeof(float));

    reallocConstArrays(const_bytes);
    const_bytes = 0;

    arg3.data   = OP_consts_h + const_bytes;
    arg3.data_d = OP_consts_d + const_bytes;

    for (int d = 0; d < 1; ++d)
        ((float *)arg3.data)[d] = arg3_host_data[d];

    const_bytes += ROUND_UP(1 * sizeof(float));

    mvConstArraysToDevice(const_bytes);

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

            op_cuda_jac_1_res<<<num_blocks, block_size>>>(
                (double *)arg0.data_d,
                (float *)arg1.data_d,
                (float *)arg2.data_d,
                arg1.map_data_d,
                (float *)arg3.data_d,
                start,
                end,
                set->size + set->exec_size
            );
        }
    }

    arg3.data = (char *)arg3_host_data;

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;


}
