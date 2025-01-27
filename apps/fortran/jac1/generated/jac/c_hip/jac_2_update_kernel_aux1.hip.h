namespace op2_m_jac_2_update_main {




static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_r,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_u_sum,
    f2c::Ptr<float> _f2c_ptr_u_max
);


static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_r,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_u_sum,
    f2c::Ptr<float> _f2c_ptr_u_max
) {
    const f2c::Span<const float, 1> r{_f2c_ptr_r, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> du{_f2c_ptr_du, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u{_f2c_ptr_u, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u_sum{_f2c_ptr_u_sum, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u_max{_f2c_ptr_u_max, f2c::Extent{1, 1}};

    u(1) = u(1) + du(1) + op2_const_alpha_d * r(1);
    du(1) = 0.0f;
    u_sum(1) = u_sum(1) + f2c::pow(u(1), 2);
    u_max(1) = f2c::max(u_max(1), u(1));
}

}


extern "C" __global__ __launch_bounds__(128)
void op2_k_jac_2_update_main_wrapper(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict gbl3,
    double *__restrict gbl4,
    const int stride_gbl,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_jac_2_update_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        update(
            f2c::Ptr{dat0 + n * 1.data[0],
            f2c::Ptr{dat1 + n * 1.data[0],
            f2c::Ptr{dat2 + n * 1.data[0],
            f2c::Ptr{gbl3 + thread_id, stride_gbl}.data[0],
            f2c::Ptr{gbl4 + thread_id, stride_gbl}.data[0]
        );
    }
}


const char op2_k_jac_2_update_main_src[] = R"_op2_k(
namespace op2_m_jac_2_update_main {

static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_r,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_u_sum,
    f2c::Ptr<float> _f2c_ptr_u_max
);


static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_r,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_u_sum,
    f2c::Ptr<float> _f2c_ptr_u_max
) {
    const f2c::Span<const float, 1> r{_f2c_ptr_r, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> du{_f2c_ptr_du, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u{_f2c_ptr_u, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u_sum{_f2c_ptr_u_sum, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> u_max{_f2c_ptr_u_max, f2c::Extent{1, 1}};

    u(1) = u(1) + du(1) + op2_const_alpha_d * r(1);
    du(1) = 0.0f;
    u_sum(1) = u_sum(1) + f2c::pow(u(1), 2);
    u_max(1) = f2c::max(u_max(1), u(1));
}

}

extern "C" __global__ __launch_bounds__(128)
void op2_k_jac_2_update_main_wrapper(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict gbl3,
    double *__restrict gbl4,
    const int stride_gbl,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_jac_2_update_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        update(
            f2c::Ptr{dat0 + n * 1.data[0],
            f2c::Ptr{dat1 + n * 1.data[0],
            f2c::Ptr{dat2 + n * 1.data[0],
            f2c::Ptr{gbl3 + thread_id, stride_gbl}.data[0],
            f2c::Ptr{gbl4 + thread_id, stride_gbl}.data[0]
        );
    }
}

)_op2_k";

__global__
static void op2_k_jac_2_update_main_init_gbls(
    double *gbl3,
    double *gbl4,
    double *gbl4_ref,
    int stride
) {
    namespace kernel = op2_m_jac_2_update_main;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl3[thread_id + d * stride] = 0;
    }
    for (int d = 0; d < 1; ++d) {
        gbl4[thread_id + d * stride] = gbl4_ref[d];
    }
}

extern "C" void op2_k_jac_2_update_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    namespace kernel = op2_m_jac_2_update_main;

    int n_args = 5;
    op_arg args[5];

    op_timing2_enter_kernel("jac_2_update", "c_CUDA", "Direct");
    op_timing2_enter("Init");

    op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_jac_2_update_main_wrapper",
                                    (void *)op2_k_jac_2_update_main_wrapper,
                                    op2_k_jac_2_update_main_src);

    if (first_invocation) {
        info.add_param("op2_const_alpha_d", &alpha, &op2_const_alpha_d, &op2_const_alpha_hash);

        first_invocation = false;
    }

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_timing2_exit();
        op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg3, (double *)arg3.data);
        op_mpi_reduce(&arg4, (double *)arg4.data);

        op_mpi_set_dirtybit_cuda(n_args, args);
        op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);



    static double* gbl4_ref_d = nullptr;

    op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    op_timing2_exit();

    auto [block_limit, block_size] = info.get_launch_config(kernel_inst, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "jac_2_update"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;


    op_timing2_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];
    arg2 = args[2];
    arg3 = args[3];
    arg4 = args[4];

    op_timing2_next("Update GBL Refs");
    if (gbl4_ref_d == nullptr) {
        CUDA_SAFE_CALL(hipMalloc(&gbl4_ref_d, 1 * sizeof(double)));
    }

    CUDA_SAFE_CALL(hipMemcpyAsync(gbl4_ref_d, arg4.data, 1 * sizeof(double), hipMemcpyHostToDevice, 0));

    op_timing2_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_jac_2_update_main_init_gbls<<<max_blocks, block_size>>>(
        (double *)arg3.data_d,
        (double *)arg4.data_d,
        gbl4_ref_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(hipPeekAtLastError());

    op_timing2_exit();
    op_timing2_next("Computation");

    int start = 0;
    int end = set->size;

    op_timing2_enter("Kernel");

    int size = f2c::round32(set->size);
    void *kernel_args[] = {
        &arg0.data_d,
        &arg1.data_d,
        &arg2.data_d,
        &arg3.data_d,
        &arg4.data_d,
        &stride_gbl,
        &start,
        &end,
        &size
    };

    void *kernel_args_jit[] = {
        &arg0.data_d,
        &arg1.data_d,
        &arg2.data_d,
        &arg3.data_d,
        &arg4.data_d,
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
    op_mpi_reduce(&arg3, arg3.data);
    op_mpi_reduce(&arg4, arg4.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(hipStreamSynchronize(0));

    op_timing2_exit();
    op_timing2_exit();
}