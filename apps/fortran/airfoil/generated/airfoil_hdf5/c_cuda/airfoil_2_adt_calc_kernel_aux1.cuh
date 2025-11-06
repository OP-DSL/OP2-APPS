namespace op2_m_airfoil_2_adt_calc_m {




static __device__ void adt_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_x3,
    f2c::Ptr<const float> _f2c_ptr_x4,
    f2c::Ptr<const float> _f2c_ptr_q,
    float& adt
);


static __device__ void adt_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_x3,
    f2c::Ptr<const float> _f2c_ptr_x4,
    f2c::Ptr<const float> _f2c_ptr_q,
    float& adt
) {
    const f2c::Span<const float, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x3{_f2c_ptr_x3, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x4{_f2c_ptr_x4, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    float dx;
    float dy;
    float ri;
    float u;
    float v;
    float c;

    ri = 1.0 / q(1);
    u = ri * q(2);
    v = ri * q(3);
    c = f2c::sqrt(op2_const_gam_d * op2_const_gm1_d * (ri * q(4) - 0.5 * (f2c::pow(u, 2) + f2c::pow(v, 2))));
    dx = x2(1) - x1(1);
    dy = x2(2) - x1(2);
    adt = f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x3(1) - x2(1);
    dy = x3(2) - x2(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x4(1) - x3(1);
    dy = x4(2) - x3(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x1(1) - x4(1);
    dy = x1(2) - x4(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    adt = adt / op2_const_cfl_d;
}

}


extern "C" __global__ 
void op2_k_airfoil_2_adt_calc_m_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_airfoil_2_adt_calc_m;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        adt_calc(
            f2c::Ptr{dat0 + map0[0 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[1 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[2 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[3 * stride + n] * 2},
            f2c::Ptr{dat1 + n * 4,
            f2c::Ptr{dat2 + n * 1.data[0]
        );
    }
}


const char op2_k_airfoil_2_adt_calc_m_src[] = R"_op2_k(
namespace op2_m_airfoil_2_adt_calc_m {

static __device__ void adt_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_x3,
    f2c::Ptr<const float> _f2c_ptr_x4,
    f2c::Ptr<const float> _f2c_ptr_q,
    float& adt
);


static __device__ void adt_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_x3,
    f2c::Ptr<const float> _f2c_ptr_x4,
    f2c::Ptr<const float> _f2c_ptr_q,
    float& adt
) {
    const f2c::Span<const float, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x3{_f2c_ptr_x3, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x4{_f2c_ptr_x4, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> q{_f2c_ptr_q, f2c::Extent{1, 4}};
    float dx;
    float dy;
    float ri;
    float u;
    float v;
    float c;

    ri = 1.0 / q(1);
    u = ri * q(2);
    v = ri * q(3);
    c = f2c::sqrt(op2_const_gam_d * op2_const_gm1_d * (ri * q(4) - 0.5 * (f2c::pow(u, 2) + f2c::pow(v, 2))));
    dx = x2(1) - x1(1);
    dy = x2(2) - x1(2);
    adt = f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x3(1) - x2(1);
    dy = x3(2) - x2(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x4(1) - x3(1);
    dy = x4(2) - x3(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    dx = x1(1) - x4(1);
    dy = x1(2) - x4(2);
    adt = adt + f2c::abs(u * dy - v * dx) + c * f2c::sqrt(f2c::pow(dx, 2) + f2c::pow(dy, 2));
    adt = adt / op2_const_cfl_d;
}

}

extern "C" __global__ 
void op2_k_airfoil_2_adt_calc_m_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_airfoil_2_adt_calc_m;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        adt_calc(
            f2c::Ptr{dat0 + map0[0 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[1 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[2 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[3 * stride + n] * 2},
            f2c::Ptr{dat1 + n * 4,
            f2c::Ptr{dat2 + n * 1.data[0]
        );
    }
}

)_op2_k";


extern "C" void op2_k_airfoil_2_adt_calc_m_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    namespace kernel = op2_m_airfoil_2_adt_calc_m;

    int n_args = 6;
    op_arg args[6];

    op_timing2_enter_kernel("airfoil_2_adt_calc", "c_CUDA", "Indirect (atomics)");
    op_timing2_enter("Init");

    op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_airfoil_2_adt_calc_m_wrapper",
                                    (void *)op2_k_airfoil_2_adt_calc_m_wrapper,
                                    op2_k_airfoil_2_adt_calc_m_src);

    if (first_invocation) {
        info.add_param("op2_const_gam_d", &gam, &op2_const_gam_d, &op2_const_gam_hash);
        info.add_param("op2_const_gm1_d", &gm1, &op2_const_gm1_d, &op2_const_gm1_hash);
        info.add_param("op2_const_cfl_d", &cfl, &op2_const_cfl_d, &op2_const_cfl_hash);

        first_invocation = false;
    }

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

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




    op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    op_timing2_exit();

    std::array<int, 3> sections = {0, set->core_size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(kernel_inst, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "airfoil_2_adt_calc"));

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
    arg4 = args[4];
    arg5 = args[5];

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
                &arg4.data_d,
                &arg5.data_d,
                &arg0.map_data_d,
                &start,
                &end,
                &size
            };

            void *kernel_args_jit[] = {
                &arg0.data_d,
                &arg4.data_d,
                &arg5.data_d,
                &arg0.map_data_d,
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