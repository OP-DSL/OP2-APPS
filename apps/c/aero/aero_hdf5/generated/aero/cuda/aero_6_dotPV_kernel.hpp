
namespace op2_k6 {
__device__ inline void dotPV(const double *p, const double *v, double *c) { *c += (*p) * (*v); }
}

__global__ void op_cuda_aero_6_dotPV(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *gbl2,

    int set_size
) {
    double gbl2_local[1];
    for (int d = 0; d < 1; ++d)
        gbl2_local[d] = ZERO_double;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k6::dotPV(
            dat0 + n * 1,
            dat1 + n * 1,
            gbl2_local
        );
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl2 + blockIdx.x * 1 + d, gbl2_local[d]);
}

void op_par_loop_aero_6_dotPV(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int num_args_expanded = 3;
    op_arg args_expanded[3];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(6);

    OP_kernels[6].name = name;
    OP_kernels[6].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_6_dotPV\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    double *arg2_host_data = (double *)arg2.data;


#ifdef OP_BLOCK_SIZE_6
    int block_size = OP_BLOCK_SIZE_6;
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

    arg2.data   = OP_reduct_h + reduction_bytes;
    arg2.data_d = OP_reduct_d + reduction_bytes;

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            ((double *)arg2.data)[b * 1 + d] = ZERO_double;
    }

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));

    mvReductArraysToDevice(reduction_bytes);

    op_cuda_aero_6_dotPV<<<num_blocks, block_size, reduction_size * block_size>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        (double *)arg2.data_d,
        set->size
    );

    mvReductArraysToHost(reduction_bytes);


    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg2_host_data[d] += ((double *)arg2.data)[b * 1 + d];
    }

    arg2.data = (char *)arg2_host_data;
    op_mpi_reduce(&arg2, arg2_host_data);

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[6].time += wall_end - wall_start;


    OP_kernels[6].transfer += (float)set->size * arg0.size;
    OP_kernels[6].transfer += (float)set->size * arg1.size;
}
