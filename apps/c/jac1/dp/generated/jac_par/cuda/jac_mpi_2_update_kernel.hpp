
namespace op2_k2 {
__device__ static inline double maxfun(double a, double b) {
   return a>b ? a : b;
}

__device__ inline void update(const double *r, double *du, double *u, int *index, double *u_sum,
                   double *u_max) {
  *u += *du + alpha_d * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = maxfun(*u_max, *u);
}
}

__global__ void op_cuda_jac_mpi_2_update(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    double *gbl4,
    double *gbl5,

    int set_size
) {
    double gbl4_local[1];
    for (int d = 0; d < 1; ++d)
        gbl4_local[d] = ZERO_double;

    double gbl5_local[1];
    for (int d = 0; d < 1; ++d)
        gbl5_local[d] = gbl5[blockIdx.x * 1 + d];

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k2::update(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            &n,
            gbl4_local,
            gbl5_local
        );
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl4 + blockIdx.x * 1 + d, gbl4_local[d]);

    for (int d = 0; d < 1; ++d)
        op_reduction<5>(gbl5 + blockIdx.x * 1 + d, gbl5_local[d]);
}

void op_par_loop_jac_mpi_2_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    int num_args_expanded = 5;
    op_arg args_expanded[5];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg4;
    args_expanded[4] = arg5;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): jac_mpi_2_update\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    double *arg4_host_data = (double *)arg4.data;
    double *arg5_host_data = (double *)arg5.data;


#ifdef OP_BLOCK_SIZE_2
    int block_size = OP_BLOCK_SIZE_2;
#else
    int block_size = OP_block_size;
#endif

    int num_blocks = 200;

    int max_blocks = num_blocks;

    int reduction_bytes = 0;
    int reduction_size = 0;

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));
    reduction_size   = MAX(reduction_size, sizeof(double));
    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));
    reduction_size   = MAX(reduction_size, sizeof(double));

    reallocReductArrays(reduction_bytes);
    reduction_bytes = 0;

    arg4.data   = OP_reduct_h + reduction_bytes;
    arg4.data_d = OP_reduct_d + reduction_bytes;

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            ((double *)arg4.data)[b * 1 + d] = ZERO_double;
    }

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));
    arg5.data   = OP_reduct_h + reduction_bytes;
    arg5.data_d = OP_reduct_d + reduction_bytes;

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            ((double *)arg5.data)[b * 1 + d] = arg5_host_data[d];
    }

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));

    mvReductArraysToDevice(reduction_bytes);

    op_cuda_jac_mpi_2_update<<<num_blocks, block_size, reduction_size * block_size>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        (double *)arg2.data_d,
        (double *)arg4.data_d,
        (double *)arg5.data_d,
        set->size
    );

    mvReductArraysToHost(reduction_bytes);


    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg4_host_data[d] += ((double *)arg4.data)[b * 1 + d];
    }

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg5_host_data[d] = MAX(arg5_host_data[d], ((double *)arg5.data)[b * 1 + d]);
    }

    arg4.data = (char *)arg4_host_data;
    op_mpi_reduce(&arg4, arg4_host_data);

    arg5.data = (char *)arg5_host_data;
    op_mpi_reduce(&arg5, arg5_host_data);

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[2].time += wall_end - wall_start;


    OP_kernels[2].transfer += (float)set->size * arg0.size;
    OP_kernels[2].transfer += (float)set->size * arg1.size * 2.0f;
    OP_kernels[2].transfer += (float)set->size * arg2.size * 2.0f;
}
