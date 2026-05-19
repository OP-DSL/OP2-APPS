namespace op2_m_aero_mpi_6_dotPV {


int op2_dotPV_gbl_stride = -1;
__constant__ int op2_dotPV_gbl_stride_d;

__device__ inline void dotPV(const double *p, const double *v, double *c) { *c += (*p) * (*v); }}


extern "C" __global__ __launch_bounds__(128)
void op2_k_aero_mpi_6_dotPV_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict gbl2,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_6_dotPV::dotPV(
            dat0 + n * 1,
            dat1 + n * 1,
            gbl2 + thread_id
        );
    }
}


const char op2_k_aero_mpi_6_dotPV_src[] = R"_op2_k(
namespace op2_m_aero_mpi_6_dotPV {

__device__ inline void dotPV(const double *p, const double *v, double *c) { *c += (*p) * (*v); }}

extern "C" __global__ __launch_bounds__(128)
void op2_k_aero_mpi_6_dotPV_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict gbl2,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_6_dotPV::dotPV(
            dat0 + n * 1,
            dat1 + n * 1,
            gbl2 + thread_id
        );
    }
}

)_op2_k";

__global__
static void op2_k_aero_mpi_6_dotPV_init_gbls(
    double *gbl2,
    int stride
) {
    namespace kernel = op2_m_aero_mpi_6_dotPV;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl2[thread_id + d * stride] = 0;
    }
}

void op_par_loop_aero_mpi_6_dotPV(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    namespace kernel = op2_m_aero_mpi_6_dotPV;

    int n_args = 3;
    op_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    // op_timing2_enter_kernel("aero_mpi_6_dotPV", "c_CUDA", "Direct");
    // op_timing2_enter("Init");

    // op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_aero_mpi_6_dotPV_wrapper",
                                    (void *)op2_k_aero_mpi_6_dotPV_wrapper,
                                    op2_k_aero_mpi_6_dotPV_src);

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "aero_mpi_6_dotPV"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;

    if (first_invocation) {

        kernel::op2_dotPV_gbl_stride = block_size * max_blocks;
        info.add_param("op2_dotPV_gbl_stride_d", &kernel::op2_dotPV_gbl_stride, &kernel::op2_dotPV_gbl_stride_d);

        first_invocation = false;
    }

    // op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        // op_timing2_exit();
        // op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg2, (double *)arg2.data);

        op_mpi_set_dirtybit_cuda(n_args, args);
        // op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);




    // op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    // op_timing2_exit();


    // op_timing2_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];
    arg2 = args[2];

    // op_timing2_next("Update GBL Refs");

    // op_timing2_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_aero_mpi_6_dotPV_init_gbls<<<max_blocks, block_size>>>(
        (double *)arg2.data_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(hipPeekAtLastError());

    // op_timing2_exit();
    // op_timing2_next("Computation");

    int start = 0;
    int end = set->size;

    // op_timing2_enter("Kernel");

    int size = f2c::round32(set->size);
    void *kernel_args[] = {
        &arg0.data_d,
        &arg1.data_d,
        &arg2.data_d,
        &start,
        &end,
        &size
    };

    void *kernel_args_jit[] = {
        &arg0.data_d,
        &arg1.data_d,
        &arg2.data_d,
        &start,
        &end,
        &size
    };

    info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);

    // op_timing2_next("Process GBLs");
    exit_sync = processDeviceGbls(args, n_args, block_size * max_blocks, block_size * max_blocks);

    // op_timing2_exit();

    // op_timing2_exit();

    // op_timing2_enter("Finalise");
    op_mpi_reduce(&arg2, (double *)arg2.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(hipStreamSynchronize(0));

    // op_timing2_exit();
    // op_timing2_exit();
}