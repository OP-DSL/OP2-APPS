namespace op2_m_jac_mpi_1_res {

float op2_gbl3;

__device__ inline void res(const float *A, const float *u, float *du, const float *beta) {
  *du += (*beta) * (*A) * (*u);
}}


extern "C" __global__ __launch_bounds__(128)
void op2_k_jac_mpi_1_res_wrapper(
    const float *__restrict dat0,
    const float *__restrict dat1,
    float *__restrict dat2,
    const int *__restrict map0,
    const float gbl3,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    float arg2_0_local[1];
    for (int d = 0; d < 1; ++d)
        arg2_0_local[d] = zero_float;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_jac_mpi_1_res::res(
            dat0 + n * 1,
            dat1 + map0[1 * stride + n] * 1,
            arg2_0_local,
            &gbl3
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat2 + map0[0 * stride + n] * 1 + d, arg2_0_local[d]);
    }
}


const char op2_k_jac_mpi_1_res_src[] = R"_op2_k(
namespace op2_m_jac_mpi_1_res {

__device__ inline void res(const float *A, const float *u, float *du, const float *beta) {
  *du += (*beta) * (*A) * (*u);
}}

extern "C" __global__ __launch_bounds__(128)
void op2_k_jac_mpi_1_res_wrapper(
    const float *__restrict dat0,
    const float *__restrict dat1,
    float *__restrict dat2,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    float arg2_0_local[1];
    for (int d = 0; d < 1; ++d)
        arg2_0_local[d] = zero_float;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_jac_mpi_1_res::res(
            dat0 + n * 1,
            dat1 + map0[1 * stride + n] * 1,
            arg2_0_local,
            &op2_gbl3_d
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat2 + map0[0 * stride + n] * 1 + d, arg2_0_local[d]);
    }
}

)_op2_k";


void op_par_loop_jac_mpi_1_res(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    namespace kernel = op2_m_jac_mpi_1_res;

    int n_args = 4;
    op_arg args[4];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    // op_timing2_enter_kernel("jac_mpi_1_res", "c_CUDA", "Indirect (atomics)");
    // op_timing2_enter("Init");

    // op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_jac_mpi_1_res_wrapper",
                                    (void *)op2_k_jac_mpi_1_res_wrapper,
                                    op2_k_jac_mpi_1_res_src);

    std::array<int, 3> sections = {0, set->core_size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "jac_mpi_1_res"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);

    if (first_invocation) {
        info.add_param("op2_gbl3_d", &kernel::op2_gbl3);

        first_invocation = false;
    }

    // op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        // op_timing2_exit();
        // op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);


        op_mpi_set_dirtybit_cuda(n_args, args);
        // op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);

    kernel::op2_gbl3 = ((float *)arg3.data)[0];



    // op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    // op_timing2_exit();


    // op_timing2_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];
    arg2 = args[2];
    arg3 = args[3];

    // op_timing2_next("Update GBL Refs");


    // op_timing2_exit();
    // op_timing2_next("Computation");

    // op_timing2_enter("Kernel");

    for (int round = 1; round < sections.size(); ++round) {
        if (round == 2) {
            // op_timing2_next("MPI Wait");
            op_mpi_wait_all_grouped(n_args, args, 2);
            // op_timing2_next("Kernel");
        }

        int start = sections[round - 1];
        int end = sections[round];

        if (end - start > 0) {
            int num_blocks = (end - start + (block_size - 1)) / block_size;
            num_blocks = std::min(num_blocks, block_limit);

            int size = f2c::round32(set->size + set->exec_size);
            void *kernel_args[] = {
                &arg0.data_d,
                &arg1.data_d,
                &arg2.data_d,
                &arg1.map_data_d,
                (void *)arg3.data,
                &start,
                &end,
                &size
            };

            void *kernel_args_jit[] = {
                &arg0.data_d,
                &arg1.data_d,
                &arg2.data_d,
                &arg1.map_data_d,
                &start,
                &end,
                &size
            };

            info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);
        }

    }

    // op_timing2_exit();

    // op_timing2_exit();

    // op_timing2_enter("Finalise");

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(hipStreamSynchronize(0));

    // op_timing2_exit();
    // op_timing2_exit();
}