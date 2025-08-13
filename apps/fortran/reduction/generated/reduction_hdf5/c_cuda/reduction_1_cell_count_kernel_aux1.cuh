namespace op2_m_reduction_1_cell_count_main {




static __device__ void cell_count(
    f2c::Ptr<float> _f2c_ptr_res,
    f2c::Ptr<int> _f2c_ptr_cell_count_result
);


static __device__ void cell_count(
    f2c::Ptr<float> _f2c_ptr_res,
    f2c::Ptr<int> _f2c_ptr_cell_count_result
) {
    const f2c::Span<float, 1> res{_f2c_ptr_res, f2c::Extent{1, 4}};
    const f2c::Span<int, 1> cell_count_result{_f2c_ptr_cell_count_result, f2c::Extent{1, 1}};
    int d;

    for (d = 1; d <= 4; ++d) {
        res(d) = 0.0f;
    }
    cell_count_result = cell_count_result + 1;
}

}


extern "C" __global__ 
void op2_k_reduction_1_cell_count_main_wrapper(
    double *__restrict dat0,
    int *__restrict gbl1,
    const int stride_gbl,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_reduction_1_cell_count_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        cell_count(
            f2c::Ptr{dat0 + n * 4,
            f2c::Ptr{gbl1 + thread_id, stride_gbl}.data[0]
        );
    }
}


const char op2_k_reduction_1_cell_count_main_src[] = R"_op2_k(
namespace op2_m_reduction_1_cell_count_main {

static __device__ void cell_count(
    f2c::Ptr<float> _f2c_ptr_res,
    f2c::Ptr<int> _f2c_ptr_cell_count_result
);


static __device__ void cell_count(
    f2c::Ptr<float> _f2c_ptr_res,
    f2c::Ptr<int> _f2c_ptr_cell_count_result
) {
    const f2c::Span<float, 1> res{_f2c_ptr_res, f2c::Extent{1, 4}};
    const f2c::Span<int, 1> cell_count_result{_f2c_ptr_cell_count_result, f2c::Extent{1, 1}};
    int d;

    for (d = 1; d <= 4; ++d) {
        res(d) = 0.0f;
    }
    cell_count_result = cell_count_result + 1;
}

}

extern "C" __global__ 
void op2_k_reduction_1_cell_count_main_wrapper(
    double *__restrict dat0,
    int *__restrict gbl1,
    const int stride_gbl,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_reduction_1_cell_count_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        cell_count(
            f2c::Ptr{dat0 + n * 4,
            f2c::Ptr{gbl1 + thread_id, stride_gbl}.data[0]
        );
    }
}

)_op2_k";

__global__
static void op2_k_reduction_1_cell_count_main_init_gbls(
    int *gbl1,
    int stride
) {
    namespace kernel = op2_m_reduction_1_cell_count_main;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl1[thread_id + d * stride] = 0;
    }
}

extern "C" void op2_k_reduction_1_cell_count_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    namespace kernel = op2_m_reduction_1_cell_count_main;

    int n_args = 2;
    op_arg args[2];

    op_timing2_enter_kernel("reduction_1_cell_count", "c_CUDA", "Direct");
    op_timing2_enter("Init");

    op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_reduction_1_cell_count_main_wrapper",
                                    (void *)op2_k_reduction_1_cell_count_main_wrapper,
                                    op2_k_reduction_1_cell_count_main_src);

    if (first_invocation) {

        first_invocation = false;
    }

    args[0] = arg0;
    args[1] = arg1;

    op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_timing2_exit();
        op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg1, (int *)arg1.data);

        op_mpi_set_dirtybit_cuda(n_args, args);
        op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);




    op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    op_timing2_exit();

    auto [block_limit, block_size] = info.get_launch_config(kernel_inst, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "reduction_1_cell_count"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;


    op_timing2_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];

    op_timing2_next("Update GBL Refs");

    op_timing2_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_reduction_1_cell_count_main_init_gbls<<<max_blocks, block_size>>>(
        (int *)arg1.data_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(cudaPeekAtLastError());

    op_timing2_exit();
    op_timing2_next("Computation");

    int start = 0;
    int end = set->size;

    op_timing2_enter("Kernel");

    int size = f2c::round32(set->size);
    void *kernel_args[] = {
        &arg0.data_d,
        &arg1.data_d,
        &stride_gbl,
        &start,
        &end,
        &size
    };

    void *kernel_args_jit[] = {
        &arg0.data_d,
        &arg1.data_d,
        &stride_gbl,
        &start,
        &end,
        &size
    };

    info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);

    op_timing2_next("Process GBLs");
    exit_sync = processDeviceGbls(args, n_args, block_size * max_blocks, block_size * max_blocks);

    op_timing2_exit();

    op_timing2_exit();

    op_timing2_enter("Finalise");
    op_mpi_reduce(&arg1, arg1.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    op_timing2_exit();
    op_timing2_exit();
}