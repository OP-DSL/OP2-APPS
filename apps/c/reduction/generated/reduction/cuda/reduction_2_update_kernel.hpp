
namespace op2_k2 {
__device__ inline void update(double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}
}

__global__ void op_cuda_reduction_2_update(
    double *__restrict dat0,
    int *gbl1,

    int set_size
) {
    int gbl1_local[1];
    for (int d = 0; d < 1; ++d)
        gbl1_local[d] = ZERO_int;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k2::update(
            dat0 + n * 4,
            gbl1_local
        );
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl1 + blockIdx.x * 1 + d, gbl1_local[d]);
}

void op_par_loop_reduction_2_update(
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
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): reduction_2_update\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    int *arg1_host_data = (int *)arg1.data;


#ifdef OP_BLOCK_SIZE_2
    int block_size = OP_BLOCK_SIZE_2;
#else
    int block_size = OP_block_size;
#endif

    int num_blocks = 200;

    int max_blocks = num_blocks;

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

    op_cuda_reduction_2_update<<<num_blocks, block_size, reduction_size * block_size>>>(
        (double *)arg0.data_d,
        (int *)arg1.data_d,
        set->size
    );

    mvReductArraysToHost(reduction_bytes);


    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg1_host_data[d] += ((int *)arg1.data)[b * 1 + d];
    }

    arg1.data = (char *)arg1_host_data;
    op_mpi_reduce(&arg1, arg1_host_data);

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[2].time += wall_end - wall_start;


    OP_kernels[2].transfer += (float)set->size * arg0.size * 2.0f;
}
