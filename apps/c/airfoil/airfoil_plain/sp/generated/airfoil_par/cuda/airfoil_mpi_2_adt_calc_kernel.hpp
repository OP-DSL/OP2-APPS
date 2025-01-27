
namespace op2_k2 {
__device__ inline void adt_calc(const float *x1, const float *x2, const float *x3,
                     const float *x4, const float *q, float *adt) {
  float dx, dy, ri, u, v, c;

  ri = 1.0f / q[0];
  u = ri * q[1];
  v = ri * q[2];
  c = sqrt(gam_d * gm1_d * (ri * q[3] - 0.5f * (u * u + v * v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  *adt = (*adt) / cfl_d;
}
}

__global__ void op_cuda_airfoil_mpi_2_adt_calc(
    const float *__restrict dat0,
    const float *__restrict dat1,
    float *__restrict dat2,
    const int *__restrict map0,
    
    int start,
    int end,
    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;

        op2_k2::adt_calc(
            dat0 + map0[round32(set_size) * 0 + n] * 2,
            dat0 + map0[round32(set_size) * 1 + n] * 2,
            dat0 + map0[round32(set_size) * 2 + n] * 2,
            dat0 + map0[round32(set_size) * 3 + n] * 2,
            dat1 + n * 4,
            dat2 + n * 1
        );
    }
}

void op_par_loop_airfoil_mpi_2_adt_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    int num_args_expanded = 6;
    op_arg args_expanded[6];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): airfoil_mpi_2_adt_calc\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);



#ifdef OP_BLOCK_SIZE_2
    int block_size = OP_BLOCK_SIZE_2;
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

            op_cuda_airfoil_mpi_2_adt_calc<<<num_blocks, block_size>>>(
                (float *)arg0.data_d,
                (float *)arg4.data_d,
                (float *)arg5.data_d,
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
    OP_kernels[2].time += wall_end - wall_start;


}
