namespace op2_m_jac_1_res_main {



double op2_gbl3;

static __device__ void res(
    f2c::Ptr<const float> _f2c_ptr_a,
    f2c::Ptr<const float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<const float> _f2c_ptr_beta
);


static __device__ void res(
    f2c::Ptr<const float> _f2c_ptr_a,
    f2c::Ptr<const float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<const float> _f2c_ptr_beta
) {
    const f2c::Span<const float, 1> a{_f2c_ptr_a, f2c::Extent{1, 1}};
    const f2c::Span<const float, 1> u{_f2c_ptr_u, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> du{_f2c_ptr_du, f2c::Extent{1, 1}};
    const f2c::Span<const float, 1> beta{_f2c_ptr_beta, f2c::Extent{1, 1}};

    atomicAdd(&(du(1)), 0.0e0 + beta(1) * a(1) * u(1));
}

}


extern "C" __global__ 
void op2_k_jac_1_res_main_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    const int *__restrict map0,
    const double gbl3,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_jac_1_res_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        res(
            f2c::Ptr{dat0 + n * 1.data[0],
            f2c::Ptr{dat1 + map0[1 * stride + n] * 1}.data[0],
            f2c::Ptr{dat2 + map0[0 * stride + n] * 1}.data[0],
            gbl3
        );
    }
}


const char op2_k_jac_1_res_main_src[] = R"_op2_k(
namespace op2_m_jac_1_res_main {

static __device__ void res(
    f2c::Ptr<const float> _f2c_ptr_a,
    f2c::Ptr<const float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<const float> _f2c_ptr_beta
);


static __device__ void res(
    f2c::Ptr<const float> _f2c_ptr_a,
    f2c::Ptr<const float> _f2c_ptr_u,
    f2c::Ptr<float> _f2c_ptr_du,
    f2c::Ptr<const float> _f2c_ptr_beta
) {
    const f2c::Span<const float, 1> a{_f2c_ptr_a, f2c::Extent{1, 1}};
    const f2c::Span<const float, 1> u{_f2c_ptr_u, f2c::Extent{1, 1}};
    const f2c::Span<float, 1> du{_f2c_ptr_du, f2c::Extent{1, 1}};
    const f2c::Span<const float, 1> beta{_f2c_ptr_beta, f2c::Extent{1, 1}};

    atomicAdd(&(du(1)), 0.0e0 + beta(1) * a(1) * u(1));
}

}

extern "C" __global__ 
void op2_k_jac_1_res_main_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_jac_1_res_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        res(
            f2c::Ptr{dat0 + n * 1.data[0],
            f2c::Ptr{dat1 + map0[1 * stride + n] * 1}.data[0],
            f2c::Ptr{dat2 + map0[0 * stride + n] * 1}.data[0],
            op2_gbl3_d
        );
    }
}

)_op2_k";


extern "C" void op2_k_jac_1_res_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    namespace kernel = op2_m_jac_1_res_main;

    int n_args = 4;
    op_arg args[4];

    op_timing2_enter_kernel("jac_1_res", "c_CUDA", "Indirect (atomics)");
    op_timing2_enter("Init");

    op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_jac_1_res_main_wrapper",
                                    (void *)op2_k_jac_1_res_main_wrapper,
                                    op2_k_jac_1_res_main_src);

    if (first_invocation) {
        info.add_param("op2_gbl3_d", &kernel::op2_gbl3);

        first_invocation = false;
    }

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;

    op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_timing2_exit();
        op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);


        op_mpi_set_dirtybit_cuda(n_args, args);
        op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);

    kernel::op2_gbl3 = ((double *)arg3.data)[0];



    op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    op_timing2_exit();

    std::array<int, 3> sections = {0, set->core_size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(kernel_inst, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "jac_1_res"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);


    op_timing2_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];
    arg2 = args[2];
    arg3 = args[3];

    op_timing2_next("Update GBL Refs");


    op_timing2_exit();
    op_timing2_next("Computation");

    op_timing2_enter("Kernel");

    for (int round = 1; round < sections.size(); ++round) {
        if (round == 2) {
            op_timing2_next("MPI Wait");
            op_mpi_wait_all_grouped(n_args, args, 2);
            op_timing2_next("Kernel");
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

    op_timing2_exit();

    op_timing2_exit();

    op_timing2_enter("Finalise");

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    op_timing2_exit();
    op_timing2_exit();
}