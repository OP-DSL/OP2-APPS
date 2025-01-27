
namespace op2_k4 {
__device__ inline void spMV(double **v, const double *K, const double **p) {
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
  v[0][0] += K[0] * p[0][0];
  v[0][0] += K[1] * p[1][0];
  v[1][0] += K[1] * p[0][0];
  v[0][0] += K[2] * p[2][0];
  v[2][0] += K[2] * p[0][0];
  v[0][0] += K[3] * p[3][0];
  v[3][0] += K[3] * p[0][0];
  v[1][0] += K[4 + 1] * p[1][0];
  v[1][0] += K[4 + 2] * p[2][0];
  v[2][0] += K[4 + 2] * p[1][0];
  v[1][0] += K[4 + 3] * p[3][0];
  v[3][0] += K[4 + 3] * p[1][0];
  v[2][0] += K[8 + 2] * p[2][0];
  v[2][0] += K[8 + 3] * p[3][0];
  v[3][0] += K[8 + 3] * p[2][0];
  v[3][0] += K[15] * p[3][0];
}
}

__global__ void op_cuda_aero_mpi_4_spMV(
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

        double arg0_1_local[1];
        for (int d = 0; d < 1; ++d)
            arg0_1_local[d] = ZERO_double;

        double arg0_2_local[1];
        for (int d = 0; d < 1; ++d)
            arg0_2_local[d] = ZERO_double;

        double arg0_3_local[1];
        for (int d = 0; d < 1; ++d)
            arg0_3_local[d] = ZERO_double;

        double *arg0_vec[4];
        arg0_vec[0] = arg0_0_local;
        arg0_vec[1] = arg0_1_local;
        arg0_vec[2] = arg0_2_local;
        arg0_vec[3] = arg0_3_local;

        const double *arg2_vec[4];
        arg2_vec[0] = dat2 + map0[round32(set_size) * 0 + n] * 1;
        arg2_vec[1] = dat2 + map0[round32(set_size) * 1 + n] * 1;
        arg2_vec[2] = dat2 + map0[round32(set_size) * 2 + n] * 1;
        arg2_vec[3] = dat2 + map0[round32(set_size) * 3 + n] * 1;

        op2_k4::spMV(
            arg0_vec,
            dat1 + n * 16,
            arg2_vec
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 0 + n] * 1 + d, arg0_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 1 + n] * 1 + d, arg0_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 2 + n] * 1 + d, arg0_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[round32(set_size) * 3 + n] * 1 + d, arg0_3_local[d]);
    }
}

void op_par_loop_aero_mpi_4_spMV(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int num_args_expanded = 9;
    op_arg args_expanded[9];

    args_expanded[0] = op_arg_dat(arg0.dat, 0, arg0.map, 1, "double", 3);
    args_expanded[1] = op_arg_dat(arg0.dat, 1, arg0.map, 1, "double", 3);
    args_expanded[2] = op_arg_dat(arg0.dat, 2, arg0.map, 1, "double", 3);
    args_expanded[3] = op_arg_dat(arg0.dat, 3, arg0.map, 1, "double", 3);
    args_expanded[4] = arg1;
    args_expanded[5] = op_arg_dat(arg2.dat, 0, arg2.map, 1, "double", 0);
    args_expanded[6] = op_arg_dat(arg2.dat, 1, arg2.map, 1, "double", 0);
    args_expanded[7] = op_arg_dat(arg2.dat, 2, arg2.map, 1, "double", 0);
    args_expanded[8] = op_arg_dat(arg2.dat, 3, arg2.map, 1, "double", 0);

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(4);

    OP_kernels[4].name = name;
    OP_kernels[4].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_mpi_4_spMV\n");

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

            op_cuda_aero_mpi_4_spMV<<<num_blocks, block_size>>>(
                (double *)arg0.data_d,
                (double *)arg1.data_d,
                (double *)arg2.data_d,
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
