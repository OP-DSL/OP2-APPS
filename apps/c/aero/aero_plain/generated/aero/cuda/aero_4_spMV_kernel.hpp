namespace op2_k4 {


__device__ inline void spMV(double *v0, double *v1, double *v2, double *v3, const double *K,
                 const double *p0, const double *p1, const double *p2, const double *p3) {
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += OP2_STRIDE(K, (j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += OP2_STRIDE(K, (j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = OP2_STRIDE(K, (j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v0[0] += K[0] * p0[0];
  v0[0] += K[1] * p1[0];
  v1[0] += K[1] * p0[0];
  v0[0] += K[2] * p2[0];
  v2[0] += K[2] * p0[0];
  v0[0] += K[3] * p3[0];
  v3[0] += K[3] * p0[0];
  v1[0] += K[4 + 1] * p1[0];
  v1[0] += K[4 + 2] * p2[0];
  v2[0] += K[4 + 2] * p1[0];
  v1[0] += K[4 + 3] * p3[0];
  v3[0] += K[4 + 3] * p1[0];
  v2[0] += K[8 + 2] * p2[0];
  v2[0] += K[8 + 3] * p3[0];
  v3[0] += K[8 + 3] * p2[0];
  v3[0] += K[15] * p3[0];
}
}

__global__ void op_cuda_aero_4_spMV(
    double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
    const int *__restrict map0,
    
    int start,
    int end,
    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;
        double arg0_0_local[1];
        for (int d = 0; d < 1; ++d)
            arg0_0_local[d] = ZERO_double;

        double arg1_1_local[1];
        for (int d = 0; d < 1; ++d)
            arg1_1_local[d] = ZERO_double;

        double arg2_2_local[1];
        for (int d = 0; d < 1; ++d)
            arg2_2_local[d] = ZERO_double;

        double arg3_3_local[1];
        for (int d = 0; d < 1; ++d)
            arg3_3_local[d] = ZERO_double;

        op2_k4::spMV(
            arg0_0_local,
            arg1_1_local,
            arg2_2_local,
            arg3_3_local,
            dat1 + n * 16,
            dat2 + map0[round32(set_size) * 0 + n] * 1,
            dat2 + map0[round32(set_size) * 1 + n] * 1,
            dat2 + map0[round32(set_size) * 2 + n] * 1,
            dat2 + map0[round32(set_size) * 3 + n] * 1
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 0 + n] * 1 + d, arg0_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 1 + n] * 1 + d, arg1_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 2 + n] * 1 + d, arg2_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 3 + n] * 1 + d, arg3_3_local[d]);
    }
}

void op_par_loop_aero_4_spMV(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7,
    op_arg arg8
) {
    int num_args_expanded = 9;
    op_arg args_expanded[9];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;
    args_expanded[6] = arg6;
    args_expanded[7] = arg7;
    args_expanded[8] = arg8;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(4);

    OP_kernels[4].name = name;
    OP_kernels[4].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_4_spMV\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);



#ifdef OP_BLOCK_SIZE_4
    int block_size = OP_BLOCK_SIZE_4;
#else
    int block_size = OP_block_size;
#endif

    for (int round = 0; round < 2; ++round ) {
        if (round == 1)
            op_mpi_wait_all_grouped(num_args_expanded, args_expanded, 2);

        int start = round == 0 ? 0 : set->core_size;
        int end = round == 0 ? set->core_size : set->size + set->exec_size;

        if (end - start > 0) {
            int num_blocks = (end - start - 1) / block_size + 1;

            op_cuda_aero_4_spMV<<<num_blocks, block_size>>>(
                (double *)arg0.data_d,
                (double *)arg4.data_d,
                (double *)arg5.data_d,
                arg0.map_data_d,
                start,
                end,
                set->size + set->exec_size
            );
        }
    }

    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[4].time += wall_end - wall_start;


}
