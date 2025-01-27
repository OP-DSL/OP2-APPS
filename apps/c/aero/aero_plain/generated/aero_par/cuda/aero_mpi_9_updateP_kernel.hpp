
namespace op2_k9 {
__device__ inline void updateP(const double *r, double *p, const double *beta) {
  *p = (*beta) * (*p) + (*r);
}
}

__global__ void op_cuda_aero_mpi_9_updateP(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *gbl2,

    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k9::updateP(
            dat0 + n * 1,
            dat1 + n * 1,
            gbl2
        );
    }
}

void op_par_loop_aero_mpi_9_updateP(
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
    op_timing_realloc(9);

    OP_kernels[9].name = name;
    OP_kernels[9].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_9_updateP\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    double *arg2_host_data = (double *)arg2.data;

    int const_bytes = 0;

    const_bytes += ROUND_UP(1 * sizeof(double));

    reallocConstArrays(const_bytes);
    const_bytes = 0;

    arg2.data   = OP_consts_h + const_bytes;
    arg2.data_d = OP_consts_d + const_bytes;

    for (int d = 0; d < 1; ++d)
        ((double *)arg2.data)[d] = arg2_host_data[d];

    const_bytes += ROUND_UP(1 * sizeof(double));

    mvConstArraysToDevice(const_bytes);

#ifdef OP_BLOCK_SIZE_9
    int block_size = OP_BLOCK_SIZE_9;
#else
    int block_size = OP_block_size;
#endif

    int num_blocks = 200;

    op_cuda_aero_mpi_9_updateP<<<num_blocks, block_size>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        (double *)arg2.data_d,
        set->size
    );



    arg2.data = (char *)arg2_host_data;

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[9].time += wall_end - wall_start;


    OP_kernels[9].transfer += (float)set->size * arg0.size;
    OP_kernels[9].transfer += (float)set->size * arg1.size * 2.0f;
}
