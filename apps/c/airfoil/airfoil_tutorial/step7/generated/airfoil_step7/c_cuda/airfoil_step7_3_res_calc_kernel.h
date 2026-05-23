namespace op2_m_airfoil_step7_3_res_calc {


__device__ inline void res_calc(const double *x1, const double *x2, const double *q1,
                     const double *q2, const double *adt1, const double *adt2,
                     double *res1, double *res2) {
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = op2_const_gm1_d * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));
  vol1 = ri * (q1[1] * dy - q1[2] * dx);

  ri = 1.0f / q2[0];
  p2 = op2_const_gm1_d * (q2[3] - 0.5f * ri * (q2[1] * q2[1] + q2[2] * q2[2]));
  vol2 = ri * (q2[1] * dy - q2[2] * dx);

  mu = 0.5f * ((*adt1) + (*adt2)) * op2_const_eps_d;

  f = 0.5f * (vol1 * q1[0] + vol2 * q2[0]) + mu * (q1[0] - q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * q2[1] + p2 * dy) +
      mu * (q1[1] - q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * q2[2] - p2 * dx) +
      mu * (q1[2] - q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (q2[3] + p2)) + mu * (q1[3] - q2[3]);
  res1[3] += f;
  res2[3] -= f;
}}


extern "C" __global__ 
void op2_k_airfoil_step7_3_res_calc_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict map0,
    const int *__restrict map1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    double arg6_0_local[4];
    for (int d = 0; d < 4; ++d)
        arg6_0_local[d] = zero_double;

    double arg7_1_local[4];
    for (int d = 0; d < 4; ++d)
        arg7_1_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_airfoil_step7_3_res_calc::res_calc(
            dat0 + map0[0 * stride + n] * 2,
            dat0 + map0[1 * stride + n] * 2,
            dat1 + map1[0 * stride + n] * 4,
            dat1 + map1[1 * stride + n] * 4,
            dat2 + map1[0 * stride + n] * 1,
            dat2 + map1[1 * stride + n] * 1,
            arg6_0_local,
            arg7_1_local
        );

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat3 + map1[0 * stride + n] * 4 + d, arg6_0_local[d]);

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat3 + map1[1 * stride + n] * 4 + d, arg7_1_local[d]);
    }
}


const char op2_k_airfoil_step7_3_res_calc_src[] = R"_op2_k(
namespace op2_m_airfoil_step7_3_res_calc {

__device__ inline void res_calc(const double *x1, const double *x2, const double *q1,
                     const double *q2, const double *adt1, const double *adt2,
                     double *res1, double *res2) {
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = op2_const_gm1_d * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));
  vol1 = ri * (q1[1] * dy - q1[2] * dx);

  ri = 1.0f / q2[0];
  p2 = op2_const_gm1_d * (q2[3] - 0.5f * ri * (q2[1] * q2[1] + q2[2] * q2[2]));
  vol2 = ri * (q2[1] * dy - q2[2] * dx);

  mu = 0.5f * ((*adt1) + (*adt2)) * op2_const_eps_d;

  f = 0.5f * (vol1 * q1[0] + vol2 * q2[0]) + mu * (q1[0] - q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * q2[1] + p2 * dy) +
      mu * (q1[1] - q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * q2[2] - p2 * dx) +
      mu * (q1[2] - q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (q2[3] + p2)) + mu * (q1[3] - q2[3]);
  res1[3] += f;
  res2[3] -= f;
}}

extern "C" __global__ 
void op2_k_airfoil_step7_3_res_calc_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict map0,
    const int *__restrict map1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    double arg6_0_local[4];
    for (int d = 0; d < 4; ++d)
        arg6_0_local[d] = zero_double;

    double arg7_1_local[4];
    for (int d = 0; d < 4; ++d)
        arg7_1_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_airfoil_step7_3_res_calc::res_calc(
            dat0 + map0[0 * stride + n] * 2,
            dat0 + map0[1 * stride + n] * 2,
            dat1 + map1[0 * stride + n] * 4,
            dat1 + map1[1 * stride + n] * 4,
            dat2 + map1[0 * stride + n] * 1,
            dat2 + map1[1 * stride + n] * 1,
            arg6_0_local,
            arg7_1_local
        );

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat3 + map1[0 * stride + n] * 4 + d, arg6_0_local[d]);

        for (int d = 0; d < 4; ++d)
            atomicAdd(dat3 + map1[1 * stride + n] * 4 + d, arg7_1_local[d]);
    }
}

)_op2_k";


void op_par_loop_airfoil_step7_3_res_calc(
    const char *name,
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
    namespace kernel = op2_m_airfoil_step7_3_res_calc;

    int n_args = 8;
    op_arg args[8];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;

    op_profile_enter_kernel("airfoil_step7_3_res_calc", "c_CUDA", "Indirect (atomics)");
    op_profile_enter("Init");

    op_profile_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_airfoil_step7_3_res_calc_wrapper",
                                    (void *)op2_k_airfoil_step7_3_res_calc_wrapper,
                                    op2_k_airfoil_step7_3_res_calc_src);

    std::array<int, 3> sections = {0, set->core_size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "airfoil_step7_3_res_calc"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);

    if (first_invocation) {
        info.add_param("op2_const_eps_d", &eps, &op2_const_eps_d, &op2_const_eps_hash);
        info.add_param("op2_const_gm1_d", &gm1, &op2_const_gm1_d, &op2_const_gm1_hash);

        first_invocation = false;
    }

    op_profile_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_profile_exit();
        op_profile_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);


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
    arg2 = args[2];
    arg3 = args[3];
    arg4 = args[4];
    arg5 = args[5];
    arg6 = args[6];
    arg7 = args[7];

    op_profile_next("Update GBL Refs");


    op_profile_exit();
    op_profile_next("Computation");

    op_profile_enter("Kernel");

    for (int round = 1; round < sections.size(); ++round) {
        if (round == 2) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all_grouped(n_args, args, 2);
            op_profile_next("Kernel");
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
                &arg4.data_d,
                &arg6.data_d,
                &arg0.map_data_d,
                &arg2.map_data_d,
                &start,
                &end,
                &size
            };

            void *kernel_args_jit[] = {
                &arg0.data_d,
                &arg2.data_d,
                &arg4.data_d,
                &arg6.data_d,
                &arg0.map_data_d,
                &arg2.map_data_d,
                &start,
                &end,
                &size
            };

            info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);
        }

    }

    op_profile_exit();

    op_profile_exit();

    op_profile_enter("Finalise");

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    op_profile_exit();
    op_profile_exit();
}