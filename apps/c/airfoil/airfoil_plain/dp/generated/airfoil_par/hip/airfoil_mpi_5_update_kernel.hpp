namespace op2_k5 {


__device__ inline void update(const double *qold, double *q, double *res,
                   const double *adt, double *rms, double *maxerr, int *idx, int *errloc) {
  double del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    double sqdel = del * del;
    *rms += sqdel;

    if (sqdel > *maxerr) {
      *maxerr = sqdel;
      *errloc = *idx;
    }
  }
}
}

__global__ void op_hip_airfoil_mpi_5_update(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    const double *__restrict dat3,
    double *__restrict gbl4,
    double *__restrict gbl5,
    int *__restrict info7,

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
        int idx = n + 1;
        int dummy_info_7 = 0; // Remove once op_arg_info functionality is fixed
        op2_k5::update(
            dat0 + n * 4,
            dat1 + n * 4,
            dat2 + n * 4,
            dat3 + n * 1,
            gbl4_local,
            gbl5_local,
            &idx,
            &dummy_info_7
        );
    }

    for (int d = 0; d < 1; ++d)
        op_reduction<3>(gbl4 + blockIdx.x * 1 + d, gbl4_local[d]);

    for (int d = 0; d < 1; ++d)
        op_reduction<5>(gbl5 + blockIdx.x * 1 + d, gbl5_local[d]);
}

void op_par_loop_airfoil_mpi_5_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7
) {
    int num_args_expanded = 7;
    op_arg args_expanded[7];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;
    args_expanded[6] = arg7;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(5);

    OP_kernels[5].name = name;
    OP_kernels[5].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): airfoil_mpi_5_update\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);


    double *arg4_host_data = (double *)arg4.data;
    double *arg5_host_data = (double *)arg5.data;


#ifdef OP_BLOCK_SIZE_5
    int block_size = OP_BLOCK_SIZE_5;
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

    op_hip_airfoil_mpi_5_update<<<num_blocks, block_size, reduction_size * block_size>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        (double *)arg2.data_d,
        (double *)arg3.data_d,
        (double *)arg4.data_d,
        (double *)arg5.data_d,
        (int *)arg7.data_d,
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
    cutilSafeCall(hipDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[5].time += wall_end - wall_start;


    OP_kernels[5].transfer += (float)set->size * arg0.size;
    OP_kernels[5].transfer += (float)set->size * arg1.size * 2.0f;
    OP_kernels[5].transfer += (float)set->size * arg2.size * 2.0f;
    OP_kernels[5].transfer += (float)set->size * arg3.size;
    OP_kernels[5].transfer += (float)set->size * arg7.size * 2.0f;
}
