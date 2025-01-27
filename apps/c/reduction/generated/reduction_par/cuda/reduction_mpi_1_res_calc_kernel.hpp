
namespace op2_k1 {
__device__ inline void res_calc(double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}
}

__global__ void op_cuda_reduction_mpi_1_res_calc(
    double *__restrict dat0,
    const int *__restrict map0,
    int *gbl1,
    
    int start,
    int end,
    int set_size
) {
    int gbl1_local[1];
    for (int d = 0; d < 1; ++d)
        gbl1_local[d] = ZERO_int;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;

        double arg0_0_local[4];
        for (int d = 0; d < 4; ++d)
            arg0_0_local[d] = ZERO_double;

        op2_k1::res_calc(
            arg0_0_local,
            gbl1_local
        );

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 0 + n] * 4 + d, arg0_0_local[d]);
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl1 + blockIdx.x * 1 + d, gbl1_local[d]);
}

void op_par_loop_reduction_mpi_1_res_calc(
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
        printf(" kernel routine (indirect): reduction_mpi_1_res_calc\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    int *arg1_host_data = (int *)arg1.data;


#ifdef OP_BLOCK_SIZE_1
    int block_size = OP_BLOCK_SIZE_1;
#else
    int block_size = OP_block_size;
#endif

    int max_blocks = (MAX(set->core_size, set->size + set->exec_size - set->core_size) - 1) / block_size + 1;

    int reduction_bytes = 0;
    int reduction_size = 0;

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(int));
    reduction_size   = MAX(reduction_size, sizeof(int));

    reallocReductArrays(reduction_bytes);
    reduction_bytes = 0;

    arg1.data   = OP_reduct_h + reduction_bytes;
    arg1.data_d = OP_reduct_d + reduction_bytes;

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            ((int *)arg1.data)[b * 1 + d] = ZERO_int;
    }

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(int));

    mvReductArraysToDevice(reduction_bytes);

    for (int round = 0; round < 3; ++round ) {
        if (round == 1)
            op_mpi_wait_all_grouped(num_args_expanded, args_expanded, 2);

        int start = round == 0 ? 0 : (round == 1 ? set->core_size : set->size);
        int end = round == 0 ? set->core_size : (round == 1 ? set->size : set->size + set->exec_size);

        if (end - start > 0) {
            int num_blocks = (end - start - 1) / block_size + 1;

            op_cuda_reduction_mpi_1_res_calc<<<num_blocks, block_size, reduction_size * block_size>>>(
                (double *)arg0.data_d,
                arg0.map_data_d,
                (int *)arg1.data_d,
                start,
                end,
                set->size + set->exec_size
            );
        }

        if (round == 1)
            mvReductArraysToHost(reduction_bytes);
    }

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg1_host_data[d] += ((int *)arg1.data)[b * 1 + d];
    }

    arg1.data = (char *)arg1_host_data;
    op_mpi_reduce(&arg1, arg1_host_data);

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;


}
