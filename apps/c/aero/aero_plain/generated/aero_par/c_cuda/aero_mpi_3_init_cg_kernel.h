namespace op2_m_aero_mpi_3_init_cg {


int op2_init_cg_gbl_stride = -1;
__constant__ int op2_init_cg_gbl_stride_d;

__device__ inline void init_cg(const double *r, double *c, double *u, double *v, double *p) {
  *c += (*r) * (*r);
  *p = *r;
  *u = 0;
  *v = 0;
}}


extern "C" __global__ 
void op2_k_aero_mpi_3_init_cg_wrapper(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict dat3,
    double *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_3_init_cg::init_cg(
            dat0 + n * 1,
            gbl1 + thread_id,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1
        );
    }
}


const char op2_k_aero_mpi_3_init_cg_src[] = R"_op2_k(
namespace op2_m_aero_mpi_3_init_cg {

__device__ inline void init_cg(const double *r, double *c, double *u, double *v, double *p) {
  *c += (*r) * (*r);
  *p = *r;
  *u = 0;
  *v = 0;
}}

extern "C" __global__ 
void op2_k_aero_mpi_3_init_cg_wrapper(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict dat3,
    double *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_3_init_cg::init_cg(
            dat0 + n * 1,
            gbl1 + thread_id,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1
        );
    }
}

)_op2_k";

__global__
static void op2_k_aero_mpi_3_init_cg_init_gbls(
    double *gbl1,
    int stride
) {
    namespace kernel = op2_m_aero_mpi_3_init_cg;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl1[thread_id + d * stride] = 0;
    }
}

void op_par_loop_aero_mpi_3_init_cg(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    namespace kernel = op2_m_aero_mpi_3_init_cg;

    int n_args = 5;
    op_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    // op_timing2_enter_kernel("aero_mpi_3_init_cg", "c_CUDA", "Direct");
    // op_timing2_enter("Init");

    // op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_aero_mpi_3_init_cg_wrapper",
                                    (void *)op2_k_aero_mpi_3_init_cg_wrapper,
                                    op2_k_aero_mpi_3_init_cg_src);

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "aero_mpi_3_init_cg"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;

    if (first_invocation) {

        kernel::op2_init_cg_gbl_stride = block_size * max_blocks;
        info.add_param("op2_init_cg_gbl_stride_d", &kernel::op2_init_cg_gbl_stride, &kernel::op2_init_cg_gbl_stride_d);

        first_invocation = false;
    }

    // op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        // op_timing2_exit();
        // op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg1, (double *)arg1.data);

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
    arg3 = args[3];
    arg4 = args[4];

    // op_timing2_next("Update GBL Refs");

    // op_timing2_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_aero_mpi_3_init_cg_init_gbls<<<max_blocks, block_size>>>(
        (double *)arg1.data_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // op_timing2_exit();
    // op_timing2_next("Computation");

    int start = 0;
    int end = set->size;

    // op_timing2_enter("Kernel");

    int size = f2c::round32(set->size);
    void *kernel_args[] = {
        &arg0.data_d,
        &arg2.data_d,
        &arg3.data_d,
        &arg4.data_d,
        &arg1.data_d,
        &start,
        &end,
        &size
    };

    void *kernel_args_jit[] = {
        &arg0.data_d,
        &arg2.data_d,
        &arg3.data_d,
        &arg4.data_d,
        &arg1.data_d,
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
    op_mpi_reduce(&arg1, (double *)arg1.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    // op_timing2_exit();
    // op_timing2_exit();
}