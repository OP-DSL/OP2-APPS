
namespace op2_k7 {
__device__ inline void updateUR(double *u, double *r, const double *p, double *v,
                     const double *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}
}

__global__ void op_cuda_aero_mpi_7_updateUR(
    double *__restrict dat0,
    double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    double *gbl4,

    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k7::updateUR(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1,
            gbl4
        );
    }
}

void op_par_loop_aero_mpi_7_updateUR(
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
    op_timing_realloc(7);

    OP_kernels[7].name = name;
    OP_kernels[7].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_7_updateUR\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    double *arg4_host_data = (double *)arg4.data;

    int const_bytes = 0;

    const_bytes += ROUND_UP(1 * sizeof(double));

    reallocConstArrays(const_bytes);
    const_bytes = 0;

    arg4.data   = OP_consts_h + const_bytes;
    arg4.data_d = OP_consts_d + const_bytes;

    for (int d = 0; d < 1; ++d)
        ((double *)arg4.data)[d] = arg4_host_data[d];

    const_bytes += ROUND_UP(1 * sizeof(double));

    mvConstArraysToDevice(const_bytes);

#ifdef OP_BLOCK_SIZE_7
    int block_size = OP_BLOCK_SIZE_7;
#else
    int block_size = OP_block_size;
#endif

    int num_blocks = 200;

    op_cuda_aero_mpi_7_updateUR<<<num_blocks, block_size>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        (double *)arg2.data_d,
        (double *)arg3.data_d,
        (double *)arg4.data_d,
        set->size
    );



    arg4.data = (char *)arg4_host_data;

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[7].time += wall_end - wall_start;


    OP_kernels[7].transfer += (float)set->size * arg0.size * 2.0f;
    OP_kernels[7].transfer += (float)set->size * arg1.size * 2.0f;
    OP_kernels[7].transfer += (float)set->size * arg2.size;
    OP_kernels[7].transfer += (float)set->size * arg3.size * 2.0f;
}
