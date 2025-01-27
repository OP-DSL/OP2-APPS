
namespace op2_k5 {
__device__ inline void update(const float *qold, float *q, float *res, const float *adt,
                   float *rms) {
  float del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    *rms += del * del;
  }
}
}

__global__ void op_cuda_airfoil_mpi_5_update(
    const float *__restrict dat0,
    float *__restrict dat1,
    float *__restrict dat2,
    const float *__restrict dat3,
    float *gbl4,

    int set_size
) {
    float gbl4_local[1];
    for (int d = 0; d < 1; ++d)
        gbl4_local[d] = ZERO_float;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k5::update(
            dat0 + n * 4,
            dat1 + n * 4,
            dat2 + n * 4,
            dat3 + n * 1,
            gbl4_local
        );
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl4 + blockIdx.x * 1 + d, gbl4_local[d]);
}

void op_par_loop_airfoil_mpi_5_update(
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
    op_timing_realloc(5);

    OP_kernels[5].name = name;
    OP_kernels[5].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): airfoil_mpi_5_update\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    float *arg4_host_data = (float *)arg4.data;


#ifdef OP_BLOCK_SIZE_5
    int block_size = OP_BLOCK_SIZE_5;
#else
    int block_size = OP_block_size;
#endif

    int num_blocks = 200;

    int max_blocks = num_blocks;

    int reduction_bytes = 0;
    int reduction_size = 0;

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(float));
    reduction_size   = MAX(reduction_size, sizeof(float));

    reallocReductArrays(reduction_bytes);
    reduction_bytes = 0;

    arg4.data   = OP_reduct_h + reduction_bytes;
    arg4.data_d = OP_reduct_d + reduction_bytes;

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            ((float *)arg4.data)[b * 1 + d] = ZERO_float;
    }

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(float));

    mvReductArraysToDevice(reduction_bytes);

    op_cuda_airfoil_mpi_5_update<<<num_blocks, block_size, reduction_size * block_size>>>(
        (float *)arg0.data_d,
        (float *)arg1.data_d,
        (float *)arg2.data_d,
        (float *)arg3.data_d,
        (float *)arg4.data_d,
        set->size
    );

    mvReductArraysToHost(reduction_bytes);


    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg4_host_data[d] += ((float *)arg4.data)[b * 1 + d];
    }

    arg4.data = (char *)arg4_host_data;
    op_mpi_reduce(&arg4, arg4_host_data);

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[5].time += wall_end - wall_start;


    OP_kernels[5].transfer += (float)set->size * arg0.size;
    OP_kernels[5].transfer += (float)set->size * arg1.size * 2.0f;
    OP_kernels[5].transfer += (float)set->size * arg2.size * 2.0f;
    OP_kernels[5].transfer += (float)set->size * arg3.size;
}
