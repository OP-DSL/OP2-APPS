namespace op2_m_airfoil_5_update_main {




static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_qold,
    f2c::Ptr<float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_res,
    const float adt,
    f2c::Ptr<float> _f2c_ptr_rms,
    float& maxerr,
    const int idx,
    int& errloc
);


static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_qold,
    f2c::Ptr<float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_res,
    const float adt,
    f2c::Ptr<float> _f2c_ptr_rms,
    float& maxerr,
    const int idx,
    int& errloc
) {
    const f2c::Span<const float, 1> qold{_f2c_ptr_qold, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res{_f2c_ptr_res, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> rms{_f2c_ptr_rms, f2c::Extent{1, 2}};
    float del;
    float adti;
    int i;

    adti = 1.0 / adt;
    for (i = 1; i <= 4; ++i) {
        del = adti * res(i);
        q(i) = qold(i) - del;
        res(i) = 0.0;
        rms(2) = rms(2) + f2c::pow(del, 2);
        if (f2c::pow(del, 2) > maxerr) {
            maxerr = f2c::pow(del, 2);
            errloc = idx;
        }
    }
}

}


extern "C" __global__ 
void op2_k_airfoil_5_update_main_wrapper(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    const double *__restrict dat3,
    double *__restrict gbl4,
    double *__restrict gbl5,
    int *__restrict info7,
    const int stride_gbl,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_airfoil_5_update_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;

        int idx = n + 1;


        update(
            f2c::Ptr{dat0 + n * 4,
            f2c::Ptr{dat1 + n * 4,
            f2c::Ptr{dat2 + n * 4,
            f2c::Ptr{dat3 + n * 1.data[0],
            f2c::Ptr{gbl4 + thread_id, stride_gbl},
            f2c::Ptr{gbl5 + thread_id, stride_gbl}.data[0],
            idx,
            f2c::Ptr{info7 + thread_id, stride_gbl}.data[0]
        );
    }
}


const char op2_k_airfoil_5_update_main_src[] = R"_op2_k(
namespace op2_m_airfoil_5_update_main {

static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_qold,
    f2c::Ptr<float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_res,
    const float adt,
    f2c::Ptr<float> _f2c_ptr_rms,
    float& maxerr,
    const int idx,
    int& errloc
);


static __device__ void update(
    f2c::Ptr<const float> _f2c_ptr_qold,
    f2c::Ptr<float> _f2c_ptr_q,
    f2c::Ptr<float> _f2c_ptr_res,
    const float adt,
    f2c::Ptr<float> _f2c_ptr_rms,
    float& maxerr,
    const int idx,
    int& errloc
) {
    const f2c::Span<const float, 1> qold{_f2c_ptr_qold, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res{_f2c_ptr_res, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> rms{_f2c_ptr_rms, f2c::Extent{1, 2}};
    float del;
    float adti;
    int i;

    adti = 1.0 / adt;
    for (i = 1; i <= 4; ++i) {
        del = adti * res(i);
        q(i) = qold(i) - del;
        res(i) = 0.0;
        rms(2) = rms(2) + f2c::pow(del, 2);
        if (f2c::pow(del, 2) > maxerr) {
            maxerr = f2c::pow(del, 2);
            errloc = idx;
        }
    }
}

}

extern "C" __global__ 
void op2_k_airfoil_5_update_main_wrapper(
    const double *__restrict dat0,
    double *__restrict dat1,
    double *__restrict dat2,
    const double *__restrict dat3,
    double *__restrict gbl4,
    double *__restrict gbl5,
    int *__restrict info7,
    const int stride_gbl,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_airfoil_5_update_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;

        int idx = n + 1;


        update(
            f2c::Ptr{dat0 + n * 4,
            f2c::Ptr{dat1 + n * 4,
            f2c::Ptr{dat2 + n * 4,
            f2c::Ptr{dat3 + n * 1.data[0],
            f2c::Ptr{gbl4 + thread_id, stride_gbl},
            f2c::Ptr{gbl5 + thread_id, stride_gbl}.data[0],
            idx,
            f2c::Ptr{info7 + thread_id, stride_gbl}.data[0]
        );
    }
}

)_op2_k";

__global__
static void op2_k_airfoil_5_update_main_init_gbls(
    double *gbl4,
    double *gbl5,
    double *gbl5_ref,
    int stride
) {
    namespace kernel = op2_m_airfoil_5_update_main;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 2; ++d) {
        gbl4[thread_id + d * stride] = 0;
    }
    for (int d = 0; d < 1; ++d) {
        gbl5[thread_id + d * stride] = gbl5_ref[d];
    }
}

extern "C" void op2_k_airfoil_5_update_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7
) {
    namespace kernel = op2_m_airfoil_5_update_main;

    int n_args = 8;
    op_arg args[8];

    op_timing2_enter_kernel("airfoil_5_update", "c_CUDA", "Direct");
    op_timing2_enter("Init");

    op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_airfoil_5_update_main_wrapper",
                                    (void *)op2_k_airfoil_5_update_main_wrapper,
                                    op2_k_airfoil_5_update_main_src);

    if (first_invocation) {

        first_invocation = false;
    }

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

    op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_timing2_exit();
        op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg4, (double *)arg4.data);
        op_mpi_reduce(&arg5, (double *)arg5.data);

        op_mpi_set_dirtybit_cuda(n_args, args);
        op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);



    static double* gbl5_ref_d = nullptr;

    op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    op_timing2_exit();

    auto [block_limit, block_size] = info.get_launch_config(kernel_inst, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "airfoil_5_update"));

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
    arg5 = args[5];
    arg6 = args[6];
    arg7 = args[7];

    op_timing2_next("Update GBL Refs");
    if (gbl5_ref_d == nullptr) {
        CUDA_SAFE_CALL(cudaMalloc(&gbl5_ref_d, 1 * sizeof(double)));
    }

    CUDA_SAFE_CALL(cudaMemcpyAsync(gbl5_ref_d, arg5.data, 1 * sizeof(double), cudaMemcpyHostToDevice, 0));

    op_timing2_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_airfoil_5_update_main_init_gbls<<<max_blocks, block_size>>>(
        (double *)arg4.data_d,
        (double *)arg5.data_d,
        gbl5_ref_d,
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
        &arg2.data_d,
        &arg3.data_d,
        &arg4.data_d,
        &arg5.data_d,
        &arg7.data_d,
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
        &arg5.data_d,
        &arg7.data_d,
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
    op_mpi_reduce(&arg4, arg4.data);
    op_mpi_reduce(&arg5, arg5.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    op_timing2_exit();
    op_timing2_exit();
}