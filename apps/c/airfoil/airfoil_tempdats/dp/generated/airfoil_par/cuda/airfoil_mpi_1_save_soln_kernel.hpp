
namespace op2_k1 {
__device__ inline void save_soln(const double *q, double *qold) {
  for (int n = 0; n < 4; n++)
    qold[n] = q[n];
}
}

__global__ void op_cuda_airfoil_mpi_1_save_soln(
    const double *__restrict dat0,
    double *__restrict dat1,

    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int n = thread_id; n < set_size; n += blockDim.x * gridDim.x) {
        op2_k1::save_soln(
            dat0 + n * 4,
            dat1 + n * 4
        );
    }
}

void op_par_loop_airfoil_mpi_1_save_soln(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    int num_args_expanded = 2;
    op_arg args_expanded[2];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): airfoil_mpi_1_save_soln\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);



#ifdef OP_BLOCK_SIZE_1
    int block_size = OP_BLOCK_SIZE_1;
#else
    int block_size = OP_block_size;
#endif

    int num_blocks = 200;

    op_cuda_airfoil_mpi_1_save_soln<<<num_blocks, block_size>>>(
        (double *)arg0.data_d,
        (double *)arg1.data_d,
        set->size
    );



    op_mpi_set_dirtybit_cuda(num_args_expanded, args_expanded);
    cutilSafeCall(cudaDeviceSynchronize());

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;


    OP_kernels[1].transfer += (float)set->size * arg0.size;
    OP_kernels[1].transfer += (float)set->size * arg1.size * 2.0f;
}
