namespace op2_m_airfoil_4_bres_calc_main {




static __device__ void bres_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    const float adt1,
    f2c::Ptr<float> _f2c_ptr_res1,
    const int bound
);


static __device__ void bres_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    const float adt1,
    f2c::Ptr<float> _f2c_ptr_res1,
    const int bound
) {
    const f2c::Span<const float, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> q1{_f2c_ptr_q1, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res1{_f2c_ptr_res1, f2c::Extent{1, 4}};
    float dx;
    float dy;
    float mu;
    float ri;
    float p1;
    float vol1;
    float p2;
    float vol2;
    float f;

    dx = x1(1) - x2(1);
    dy = x1(2) - x2(2);
    ri = 1.0 / q1(1);
    p1 = op2_const_gm1_d * (q1(4) - 0.5 * ri * (f2c::pow(q1(2), 2) + f2c::pow(q1(3), 2)));
    if (bound == 1) {
        atomicAdd(&(res1(2)), 0.0e0 + p1 * dy);
        atomicAdd(&(res1(3)), 0.0e0 - p1 * dx);
        return;
    }
    vol1 = ri * (q1(2) * dy - q1(3) * dx);
    ri = 1.0 / op2_const_qinf_d[(1) - 1];
    p2 = op2_const_gm1_d * (op2_const_qinf_d[(4) - 1] - 0.5 * ri * (f2c::pow(op2_const_qinf_d[(2) - 1], 2) + f2c::pow(op2_const_qinf_d[(3) - 1], 2)));
    vol2 = ri * (op2_const_qinf_d[(2) - 1] * dy - op2_const_qinf_d[(3) - 1] * dx);
    mu = adt1 * op2_const_eps_d;
    f = 0.5 * (vol1 * q1(1) + vol2 * op2_const_qinf_d[(1) - 1]) + mu * (q1(1) - op2_const_qinf_d[(1) - 1]);
    atomicAdd(&(res1(1)), 0.0e0 + f);
    f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * op2_const_qinf_d[(2) - 1] + p2 * dy) + mu * (q1(2) - op2_const_qinf_d[(2) - 1]);
    atomicAdd(&(res1(2)), 0.0e0 + f);
    f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * op2_const_qinf_d[(3) - 1] - p2 * dx) + mu * (q1(3) - op2_const_qinf_d[(3) - 1]);
    atomicAdd(&(res1(3)), 0.0e0 + f);
    f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (op2_const_qinf_d[(4) - 1] + p2)) + mu * (q1(4) - op2_const_qinf_d[(4) - 1]);
    atomicAdd(&(res1(4)), 0.0e0 + f);
}

}


extern "C" __global__ 
void op2_k_airfoil_4_bres_calc_main_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict dat4,
    const int *__restrict map0,
    const int *__restrict map1,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_airfoil_4_bres_calc_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        bres_calc(
            f2c::Ptr{dat0 + map0[0 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[1 * stride + n] * 2},
            f2c::Ptr{dat1 + map1[0 * stride + n] * 4},
            f2c::Ptr{dat2 + map1[0 * stride + n] * 1}.data[0],
            f2c::Ptr{dat3 + map1[0 * stride + n] * 4},
            f2c::Ptr{dat4 + n * 1.data[0]
        );
    }
}


const char op2_k_airfoil_4_bres_calc_main_src[] = R"_op2_k(
namespace op2_m_airfoil_4_bres_calc_main {

static __device__ void bres_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    const float adt1,
    f2c::Ptr<float> _f2c_ptr_res1,
    const int bound
);


static __device__ void bres_calc(
    f2c::Ptr<const float> _f2c_ptr_x1,
    f2c::Ptr<const float> _f2c_ptr_x2,
    f2c::Ptr<const float> _f2c_ptr_q1,
    const float adt1,
    f2c::Ptr<float> _f2c_ptr_res1,
    const int bound
) {
    const f2c::Span<const float, 1> x1{_f2c_ptr_x1, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> x2{_f2c_ptr_x2, f2c::Extent{1, 2}};
    const f2c::Span<const float, 1> q1{_f2c_ptr_q1, f2c::Extent{1, 4}};
    const f2c::Span<float, 1> res1{_f2c_ptr_res1, f2c::Extent{1, 4}};
    float dx;
    float dy;
    float mu;
    float ri;
    float p1;
    float vol1;
    float p2;
    float vol2;
    float f;

    dx = x1(1) - x2(1);
    dy = x1(2) - x2(2);
    ri = 1.0 / q1(1);
    p1 = op2_const_gm1_d * (q1(4) - 0.5 * ri * (f2c::pow(q1(2), 2) + f2c::pow(q1(3), 2)));
    if (bound == 1) {
        atomicAdd(&(res1(2)), 0.0e0 + p1 * dy);
        atomicAdd(&(res1(3)), 0.0e0 - p1 * dx);
        return;
    }
    vol1 = ri * (q1(2) * dy - q1(3) * dx);
    ri = 1.0 / op2_const_qinf_d[(1) - 1];
    p2 = op2_const_gm1_d * (op2_const_qinf_d[(4) - 1] - 0.5 * ri * (f2c::pow(op2_const_qinf_d[(2) - 1], 2) + f2c::pow(op2_const_qinf_d[(3) - 1], 2)));
    vol2 = ri * (op2_const_qinf_d[(2) - 1] * dy - op2_const_qinf_d[(3) - 1] * dx);
    mu = adt1 * op2_const_eps_d;
    f = 0.5 * (vol1 * q1(1) + vol2 * op2_const_qinf_d[(1) - 1]) + mu * (q1(1) - op2_const_qinf_d[(1) - 1]);
    atomicAdd(&(res1(1)), 0.0e0 + f);
    f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * op2_const_qinf_d[(2) - 1] + p2 * dy) + mu * (q1(2) - op2_const_qinf_d[(2) - 1]);
    atomicAdd(&(res1(2)), 0.0e0 + f);
    f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * op2_const_qinf_d[(3) - 1] - p2 * dx) + mu * (q1(3) - op2_const_qinf_d[(3) - 1]);
    atomicAdd(&(res1(3)), 0.0e0 + f);
    f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (op2_const_qinf_d[(4) - 1] + p2)) + mu * (q1(4) - op2_const_qinf_d[(4) - 1]);
    atomicAdd(&(res1(4)), 0.0e0 + f);
}

}

extern "C" __global__ 
void op2_k_airfoil_4_bres_calc_main_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict dat4,
    const int *__restrict map0,
    const int *__restrict map1,
    const int start,
    const int end,
    const int stride
) {
    using namespace op2_m_airfoil_4_bres_calc_main;
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        bres_calc(
            f2c::Ptr{dat0 + map0[0 * stride + n] * 2},
            f2c::Ptr{dat0 + map0[1 * stride + n] * 2},
            f2c::Ptr{dat1 + map1[0 * stride + n] * 4},
            f2c::Ptr{dat2 + map1[0 * stride + n] * 1}.data[0],
            f2c::Ptr{dat3 + map1[0 * stride + n] * 4},
            f2c::Ptr{dat4 + n * 1.data[0]
        );
    }
}

)_op2_k";


extern "C" void op2_k_airfoil_4_bres_calc_main_c(
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    namespace kernel = op2_m_airfoil_4_bres_calc_main;

    int n_args = 6;
    op_arg args[6];

    op_timing2_enter_kernel("airfoil_4_bres_calc", "c_CUDA", "Indirect (atomics)");
    op_timing2_enter("Init");

    op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_airfoil_4_bres_calc_main_wrapper",
                                    (void *)op2_k_airfoil_4_bres_calc_main_wrapper,
                                    op2_k_airfoil_4_bres_calc_main_src);

    if (first_invocation) {
        info.add_param("op2_const_qinf_d", qinf, sizeof(op2_const_qinf_d) / sizeof(qinf[0]), op2_const_qinf_d, &op2_const_qinf_hash);
        info.add_param("op2_const_eps_d", &eps, &op2_const_eps_d, &op2_const_eps_hash);
        info.add_param("op2_const_gm1_d", &gm1, &op2_const_gm1_d, &op2_const_gm1_hash);

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
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "airfoil_4_bres_calc"));

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
                &arg2.data_d,
                &arg3.data_d,
                &arg4.data_d,
                &arg5.data_d,
                &arg0.map_data_d,
                &arg2.map_data_d,
                &start,
                &end,
                &size
            };

            void *kernel_args_jit[] = {
                &arg0.data_d,
                &arg2.data_d,
                &arg3.data_d,
                &arg4.data_d,
                &arg5.data_d,
                &arg0.map_data_d,
                &arg2.map_data_d,
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