
namespace op2_k10 {
__device__ inline void update(double *phim, double *res, const double *u, double *rms) {
  *phim -= *u;
  *res = 0.0;
  *rms += (*u) * (*u);
}
}

__global__ void op_cuda_aero_mpi_10_update(
    double *__restrict dat0,
    double *__restrict dat1,
    const double *__restrict dat2,
    double *gbl3,

    int set_size
) {
    double gbl3_local[1];
    for (int d = 0; d < 1; ++d)
        gbl3_local[d] = ZERO_double;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k10::update(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            gbl3_local
        );
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl3 + blockIdx.x * 1 + d, gbl3_local[d]);
}

void op_par_loop_aero_mpi_10_update(
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
    op_timing_realloc(10);

    OP_kernels[10].name = name;
    OP_kernels[10].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_10_update\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    double *arg3_host_data = (double *)arg3.data;


#ifdef OP_BLOCK_SIZE_10
    int block_size = OP_BLOCK_SIZE_10;
#else
    int block_size = OP_block_size;
#endif

    int num_blocks = 200;

    int max_blocks = num_blocks;

    int reduction_bytes = 0;
    int reduction_size = 0;

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));
    reduction_size   = MAX(reduction_size, sizeof(double));

    reallocReductArrays(reduction_bytes);
    reduction_bytes = 0;

    arg3.data   = OP_reduct_h + reduction_bytes;
    arg3.data_d = OP_reduct_d + reduction_bytes;

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            ((double *)arg3.data)[b * 1 + d] = ZERO_double;
    }

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));

    mvReductArraysToDevice(reduction_bytes);

    op_cuda_aero_mpi_10_update<<<num_blocks, block_size, reduction_size * block_size>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        (double *)arg2.data_d,
        (double *)arg3.data_d,
        set->size
    );

    mvReductArraysToHost(reduction_bytes);


    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg3_host_data[d] += ((double *)arg3.data)[b * 1 + d];
    }

    arg3.data = (char *)arg3_host_data;
    op_mpi_reduce(&arg3, arg3_host_data);

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[10].time += wall_end - wall_start;


    OP_kernels[10].transfer += (float)set->size * arg0.size * 2.0f;
    OP_kernels[10].transfer += (float)set->size * arg1.size * 2.0f;
    OP_kernels[10].transfer += (float)set->size * arg2.size;
}
