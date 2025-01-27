
namespace op2_k3 {
__device__ inline void init_cg(const double *r, double *c, double *u, double *v, double *p) {
  *c += (*r) * (*r);
  *p = *r;
  *u = 0;
  *v = 0;
}
}

__global__ void op_cuda_aero_3_init_cg(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict dat3,
    double *gbl1,

    int set_size
) {
    double gbl1_local[1];
    for (int d = 0; d < 1; ++d)
        gbl1_local[d] = ZERO_double;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k3::init_cg(
            dat0 + n * 1,
            gbl1_local,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1
        );
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl1 + blockIdx.x * 1 + d, gbl1_local[d]);
}

void op_par_loop_aero_3_init_cg(
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
    op_timing_realloc(3);

    OP_kernels[3].name = name;
    OP_kernels[3].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_3_init_cg\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    double *arg1_host_data = (double *)arg1.data;


#ifdef OP_BLOCK_SIZE_3
    int block_size = OP_BLOCK_SIZE_3;
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

    arg1.data   = OP_reduct_h + reduction_bytes;
    arg1.data_d = OP_reduct_d + reduction_bytes;

    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            ((double *)arg1.data)[b * 1 + d] = ZERO_double;
    }

    reduction_bytes += ROUND_UP(max_blocks * 1 * sizeof(double));

    mvReductArraysToDevice(reduction_bytes);

    op_cuda_aero_3_init_cg<<<num_blocks, block_size, reduction_size * block_size>>>(
        (double *)arg0.data_d,
        (double *)arg2.data_d,
        (double *)arg3.data_d,
        (double *)arg4.data_d,
        (double *)arg1.data_d,
        set->size
    );

    mvReductArraysToHost(reduction_bytes);


    for (int b = 0; b < max_blocks; ++b) {
        for (int d = 0; d < 1; ++d)
            arg1_host_data[d] += ((double *)arg1.data)[b * 1 + d];
    }

    arg1.data = (char *)arg1_host_data;
    op_mpi_reduce(&arg1, arg1_host_data);

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[3].time += wall_end - wall_start;


    OP_kernels[3].transfer += (float)set->size * arg0.size;
    OP_kernels[3].transfer += (float)set->size * arg2.size * 2.0f;
    OP_kernels[3].transfer += (float)set->size * arg3.size * 2.0f;
    OP_kernels[3].transfer += (float)set->size * arg4.size * 2.0f;
}
