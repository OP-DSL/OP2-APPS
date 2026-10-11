namespace op2_m_aero_8_dotR {


int op2_dotR_gbl_stride = -1;
__constant__ int op2_dotR_gbl_stride_d;

__device__ inline void dotR(const double *r, double *c) { *c += (*r) * (*r); }}


extern "C" __global__ __launch_bounds__(128)
void op2_k_aero_8_dotR_wrapper(
    const double *__restrict dat0,
    double *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_8_dotR::dotR(
            dat0 + n * 1,
            gbl1 + thread_id
        );
    }
}


const char op2_k_aero_8_dotR_src[] = R"_op2_k(
namespace op2_m_aero_8_dotR {

__device__ inline void dotR(const double *r, double *c) { *c += (*r) * (*r); }}

extern "C" __global__ __launch_bounds__(128)
void op2_k_aero_8_dotR_wrapper(
    const double *__restrict dat0,
    double *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_8_dotR::dotR(
            dat0 + n * 1,
            gbl1 + thread_id
        );
    }
}

)_op2_k";

__global__
static void op2_k_aero_8_dotR_init_gbls(
    double *gbl1,
    int stride
) {
    namespace kernel = op2_m_aero_8_dotR;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl1[thread_id + d * stride] = 0;
    }
}

void op_par_loop_aero_8_dotR(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    namespace kernel = op2_m_aero_8_dotR;

    int n_args = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    op_profile_enter_kernel("aero_8_dotR", "c_CUDA", "Direct");
    op_profile_enter("Init");

    op_profile_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_aero_8_dotR_wrapper",
                                    (void *)op2_k_aero_8_dotR_wrapper,
                                    op2_k_aero_8_dotR_src);

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "aero_8_dotR"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;

    if (first_invocation) {

        kernel::op2_dotR_gbl_stride = block_size * max_blocks;
        info.add_param("op2_dotR_gbl_stride_d", &kernel::op2_dotR_gbl_stride, &kernel::op2_dotR_gbl_stride_d);

        first_invocation = false;
    }

    op_profile_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_profile_exit();
        op_profile_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg1, (double *)arg1.data);

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
    op2_k_aero_8_dotR_init_gbls<<<max_blocks, block_size>>>(
        (double *)arg1.data_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(hipPeekAtLastError());

    op_profile_exit();
    op_profile_next("Computation");

    int start = 0;
    int end = set->size;

    op_profile_enter("Kernel");

    int size = f2c::round32(set->size);
    void *kernel_args[] = {
        &arg0.data_d,
        &arg1.data_d,
        &start,
        &end,
        &size
    };

    void *kernel_args_jit[] = {
        &arg0.data_d,
        &arg1.data_d,
        &start,
        &end,
        &size
    };

    info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);

    op_profile_next("Process GBLs");
    exit_sync = processDeviceGbls(args, n_args, block_size * max_blocks, block_size * max_blocks);

    op_profile_exit();

    op_profile_exit();

    op_profile_enter("Finalise");
    op_mpi_reduce(&arg1, (double *)arg1.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(hipStreamSynchronize(0));

    op_profile_exit();
    op_profile_exit();
}