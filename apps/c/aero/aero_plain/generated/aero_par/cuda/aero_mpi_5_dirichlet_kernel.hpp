
namespace op2_k5 {
__device__ inline void dirichlet(double *res) { *res = 0.0; }
}

__global__ void op_cuda_aero_mpi_5_dirichlet(
    double *__restrict dat0,
    const int *__restrict map0,
    
    int start,
    int end,
    int set_size
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (thread_id + start < end) {
        int n = thread_id + start;

        op2_k5::dirichlet(
            dat0 + map0[round32(set_size) * 0 + n] * 1
        );
    }
}

void op_par_loop_aero_mpi_5_dirichlet(
    const char *name,
    op_set set,
    op_arg arg0
) {
    int num_args_expanded = 1;
    op_arg args_expanded[1];

    args_expanded[0] = arg0;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(5);

    OP_kernels[5].name = name;
    OP_kernels[5].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_mpi_5_dirichlet\n");

    int set_size = op_mpi_halo_exchanges_grouped(set, num_args_expanded, args_expanded, 2);



#ifdef OP_BLOCK_SIZE_5
    int block_size = OP_BLOCK_SIZE_5;
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

            op_cuda_aero_mpi_5_dirichlet<<<num_blocks, block_size>>>(
                (double *)arg0.data_d,
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
    OP_kernels[5].time += wall_end - wall_start;


}
