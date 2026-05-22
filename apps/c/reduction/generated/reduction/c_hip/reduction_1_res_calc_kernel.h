namespace op2_m_reduction_1_res_calc {


int op2_res_calc_gbl_stride = -1;
__constant__ int op2_res_calc_gbl_stride_d;

__device__ inline void res_calc(double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}}


extern "C" __global__ __launch_bounds__(128)
void op2_k_reduction_1_res_calc_wrapper(
    double *__restrict dat0,
    const int *__restrict map0,
    int *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    double arg0_0_local[4];
    for (int d = 0; d < 4; ++d)
        arg0_0_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_reduction_1_res_calc::res_calc(
            arg0_0_local,
            gbl1 + thread_id
        );

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat0 + map0[0 * stride + n] * 4 + d, arg0_0_local[d]);
    }
}


const char op2_k_reduction_1_res_calc_src[] = R"_op2_k(
namespace op2_m_reduction_1_res_calc {

__device__ inline void res_calc(double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}}

extern "C" __global__ __launch_bounds__(128)
void op2_k_reduction_1_res_calc_wrapper(
    double *__restrict dat0,
    const int *__restrict map0,
    int *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    double arg0_0_local[4];
    for (int d = 0; d < 4; ++d)
        arg0_0_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_reduction_1_res_calc::res_calc(
            arg0_0_local,
            gbl1 + thread_id
        );

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat0 + map0[0 * stride + n] * 4 + d, arg0_0_local[d]);
    }
}

)_op2_k";

__global__
static void op2_k_reduction_1_res_calc_init_gbls(
    int *gbl1,
    int stride
) {
    namespace kernel = op2_m_reduction_1_res_calc;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl1[thread_id + d * stride] = 0;
    }
}

void op_par_loop_reduction_1_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    namespace kernel = op2_m_reduction_1_res_calc;

    int n_args = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    op_profile_enter_kernel("reduction_1_res_calc", "c_CUDA", "Indirect (atomics)");
    op_profile_enter("Init");

    op_profile_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_reduction_1_res_calc_wrapper",
                                    (void *)op2_k_reduction_1_res_calc_wrapper,
                                    op2_k_reduction_1_res_calc_src);

    std::array<int, 4> sections = {0, set->core_size, set->size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "reduction_1_res_calc"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);

    if (first_invocation) {

        kernel::op2_res_calc_gbl_stride = block_size * max_blocks;
        info.add_param("op2_res_calc_gbl_stride_d", &kernel::op2_res_calc_gbl_stride, &kernel::op2_res_calc_gbl_stride_d);

        first_invocation = false;
    }

    op_profile_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_profile_exit();
        op_profile_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg1, (int *)arg1.data);

        op_mpi_set_dirtybit_cuda(n_args, args);
        op_profile_exit();
        return;
    }

    setGblIncAtomic(false);




    op_profile_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    op_profile_exit();


    op_profile_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];

    op_profile_next("Update GBL Refs");

    op_profile_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_reduction_1_res_calc_init_gbls<<<max_blocks, block_size>>>(
        (int *)arg1.data_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(hipPeekAtLastError());

    op_profile_exit();
    op_profile_next("Computation");

    op_profile_enter("Kernel");

    for (int round = 1; round < sections.size(); ++round) {
        if (round == 2) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all_grouped(n_args, args, 2);
            op_profile_next("Kernel");
        }

        int start = sections[round - 1];
        int end = sections[round];

        if (end - start > 0) {
            int num_blocks = (end - start + (block_size - 1)) / block_size;
            num_blocks = std::min(num_blocks, block_limit);

            int size = f2c::round32(set->size + set->exec_size);
            void *kernel_args[] = {
                &arg0.data_d,
                &arg0.map_data_d,
                &arg1.data_d,
                &start,
                &end,
                &size
            };

            void *kernel_args_jit[] = {
                &arg0.data_d,
                &arg0.map_data_d,
                &arg1.data_d,
                &start,
                &end,
                &size
            };

            info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);
        }

        if (round == 2) {
            op_profile_next("Process GBLs");
            exit_sync = processDeviceGbls(args, n_args, block_size * max_blocks, block_size * max_blocks);
            op_profile_next("Kernel");
        }
    }

    op_profile_exit();

    op_profile_exit();

    op_profile_enter("Finalise");
    op_mpi_reduce(&arg1, (int *)arg1.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(hipStreamSynchronize(0));

    op_profile_exit();
    op_profile_exit();
}