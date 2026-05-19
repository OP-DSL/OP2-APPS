namespace op2_m_aero_mpi_7_updateUR {

double op2_gbl4;

__device__ inline void updateUR(double *u, double *r, const double *p, double *v,
                     const double *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}}


extern "C" __global__ __launch_bounds__(128)
void op2_k_aero_mpi_7_updateUR_wrapper(
    double *__restrict dat0,
    double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    const double gbl4,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_7_updateUR::updateUR(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1,
            &gbl4
        );
    }
}


const char op2_k_aero_mpi_7_updateUR_src[] = R"_op2_k(
namespace op2_m_aero_mpi_7_updateUR {

__device__ inline void updateUR(double *u, double *r, const double *p, double *v,
                     const double *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}}

extern "C" __global__ __launch_bounds__(128)
void op2_k_aero_mpi_7_updateUR_wrapper(
    double *__restrict dat0,
    double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_7_updateUR::updateUR(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1,
            &op2_gbl4_d
        );
    }
}

)_op2_k";


void op_par_loop_aero_mpi_7_updateUR(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    namespace kernel = op2_m_aero_mpi_7_updateUR;

    int n_args = 5;
    op_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    // op_timing2_enter_kernel("aero_mpi_7_updateUR", "c_CUDA", "Direct");
    // op_timing2_enter("Init");

    // op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_aero_mpi_7_updateUR_wrapper",
                                    (void *)op2_k_aero_mpi_7_updateUR_wrapper,
                                    op2_k_aero_mpi_7_updateUR_src);

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "aero_mpi_7_updateUR"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;

    if (first_invocation) {
        info.add_param("op2_gbl4_d", &kernel::op2_gbl4);

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

    kernel::op2_gbl4 = ((double *)arg4.data)[0];



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
        &arg3.data_d,
        (void *)arg4.data,
        &start,
        &end,
        &size
    };

    void *kernel_args_jit[] = {
        &arg0.data_d,
        &arg1.data_d,
        &arg2.data_d,
        &arg3.data_d,
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

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(hipStreamSynchronize(0));

    // op_timing2_exit();
    // op_timing2_exit();
}