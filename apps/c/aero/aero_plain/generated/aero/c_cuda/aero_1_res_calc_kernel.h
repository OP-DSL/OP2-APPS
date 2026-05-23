namespace op2_m_aero_1_res_calc {


__device__ inline void res_calc(const double *x0, const double *x1, const double *x2, const double *x3,
                     const double *phim0, const double *phim1, const double *phim2, const double *phim3,
                     double *K, /*double *Kt,*/ double *res0, double *res1, double *res2, double *res3) {
  double x[4][2], phim[4];
  x[0][0] = x0[0]; x[1][0] = x1[0]; x[2][0] = x2[0]; x[3][0] = x3[0];
  x[0][1] = x0[1]; x[1][1] = x1[1]; x[2][1] = x2[1]; x[3][1] = x3[1];
  phim[0] = phim0[0]; phim[1] = phim1[0]; phim[2] = phim2[0]; phim[3] = phim3[0];

  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      K[j * 4 + k] = 0;
    }
  }
  for (int i = 0; i < 4; i++) { // for each gauss point
    double det_x_xi = 0;
    double N_x[8];

    double a = 0;
    for (int m = 0; m < 4; m++)
      det_x_xi += op2_const_Ng2_xi_d[4 * i + 16 + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] = det_x_xi * op2_const_Ng2_xi_d[4 * i + m];

    a = 0;
    for (int m = 0; m < 4; m++)
      a += op2_const_Ng2_xi_d[4 * i + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] = a * op2_const_Ng2_xi_d[4 * i + 16 + m];

    det_x_xi *= a;

    a = 0;
    for (int m = 0; m < 4; m++)
      a += op2_const_Ng2_xi_d[4 * i + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] -= a * op2_const_Ng2_xi_d[4 * i + 16 + m];

    double b = 0;
    for (int m = 0; m < 4; m++)
      b += op2_const_Ng2_xi_d[4 * i + 16 + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] -= b * op2_const_Ng2_xi_d[4 * i + m];

    det_x_xi -= a * b;

    for (int j = 0; j < 8; j++)
      N_x[j] /= det_x_xi;

    double wt1 = op2_const_wtg2_d[i] * det_x_xi;
    // double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0; j < 4; j++) {
      u[0] += N_x[j] * phim[j];
      u[1] += N_x[4 + j] * phim[j];
    }

    double Dk = 1.0 + 0.5 * op2_const_gm1_d * (op2_const_m2_d - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, op2_const_gm1i_d); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    res0[0] += wt1 * rho * (u[0] * N_x[0] + u[1] * N_x[4 + 0]);
    res1[0] += wt1 * rho * (u[0] * N_x[1] + u[1] * N_x[4 + 1]);
    res2[0] += wt1 * rho * (u[0] * N_x[2] + u[1] * N_x[4 + 2]);
    res3[0] += wt1 * rho * (u[0] * N_x[3] + u[1] * N_x[4 + 3]);

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        K[j * 4 + k] +=
            wt1 * rho * (N_x[j] * N_x[k] + N_x[4 + j] * N_x[4 + k]) -
            wt1 * rc2 * (u[0] * N_x[j] + u[1] * N_x[4 + j]) *
                (u[0] * N_x[k] + u[1] * N_x[4 + k]);
      }
    }
  }
}}


extern "C" __global__ 
void op2_k_aero_1_res_calc_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    double arg9_0_local[1];
    for (int d = 0; d < 1; ++d)
        arg9_0_local[d] = zero_double;

    double arg10_1_local[1];
    for (int d = 0; d < 1; ++d)
        arg10_1_local[d] = zero_double;

    double arg11_2_local[1];
    for (int d = 0; d < 1; ++d)
        arg11_2_local[d] = zero_double;

    double arg12_3_local[1];
    for (int d = 0; d < 1; ++d)
        arg12_3_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_1_res_calc::res_calc(
            dat0 + map0[0 * stride + n] * 2,
            dat0 + map0[1 * stride + n] * 2,
            dat0 + map0[2 * stride + n] * 2,
            dat0 + map0[3 * stride + n] * 2,
            dat1 + map0[0 * stride + n] * 1,
            dat1 + map0[1 * stride + n] * 1,
            dat1 + map0[2 * stride + n] * 1,
            dat1 + map0[3 * stride + n] * 1,
            dat2 + n * 16,
            arg9_0_local,
            arg10_1_local,
            arg11_2_local,
            arg12_3_local
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[0 * stride + n] * 1 + d, arg9_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[1 * stride + n] * 1 + d, arg10_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[2 * stride + n] * 1 + d, arg11_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[3 * stride + n] * 1 + d, arg12_3_local[d]);
    }
}


const char op2_k_aero_1_res_calc_src[] = R"_op2_k(
namespace op2_m_aero_1_res_calc {

__device__ inline void res_calc(const double *x0, const double *x1, const double *x2, const double *x3,
                     const double *phim0, const double *phim1, const double *phim2, const double *phim3,
                     double *K, /*double *Kt,*/ double *res0, double *res1, double *res2, double *res3) {
  double x[4][2], phim[4];
  x[0][0] = x0[0]; x[1][0] = x1[0]; x[2][0] = x2[0]; x[3][0] = x3[0];
  x[0][1] = x0[1]; x[1][1] = x1[1]; x[2][1] = x2[1]; x[3][1] = x3[1];
  phim[0] = phim0[0]; phim[1] = phim1[0]; phim[2] = phim2[0]; phim[3] = phim3[0];

  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      K[j * 4 + k] = 0;
    }
  }
  for (int i = 0; i < 4; i++) { // for each gauss point
    double det_x_xi = 0;
    double N_x[8];

    double a = 0;
    for (int m = 0; m < 4; m++)
      det_x_xi += op2_const_Ng2_xi_d[4 * i + 16 + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] = det_x_xi * op2_const_Ng2_xi_d[4 * i + m];

    a = 0;
    for (int m = 0; m < 4; m++)
      a += op2_const_Ng2_xi_d[4 * i + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] = a * op2_const_Ng2_xi_d[4 * i + 16 + m];

    det_x_xi *= a;

    a = 0;
    for (int m = 0; m < 4; m++)
      a += op2_const_Ng2_xi_d[4 * i + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] -= a * op2_const_Ng2_xi_d[4 * i + 16 + m];

    double b = 0;
    for (int m = 0; m < 4; m++)
      b += op2_const_Ng2_xi_d[4 * i + 16 + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] -= b * op2_const_Ng2_xi_d[4 * i + m];

    det_x_xi -= a * b;

    for (int j = 0; j < 8; j++)
      N_x[j] /= det_x_xi;

    double wt1 = op2_const_wtg2_d[i] * det_x_xi;
    // double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0; j < 4; j++) {
      u[0] += N_x[j] * phim[j];
      u[1] += N_x[4 + j] * phim[j];
    }

    double Dk = 1.0 + 0.5 * op2_const_gm1_d * (op2_const_m2_d - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, op2_const_gm1i_d); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    res0[0] += wt1 * rho * (u[0] * N_x[0] + u[1] * N_x[4 + 0]);
    res1[0] += wt1 * rho * (u[0] * N_x[1] + u[1] * N_x[4 + 1]);
    res2[0] += wt1 * rho * (u[0] * N_x[2] + u[1] * N_x[4 + 2]);
    res3[0] += wt1 * rho * (u[0] * N_x[3] + u[1] * N_x[4 + 3]);

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        K[j * 4 + k] +=
            wt1 * rho * (N_x[j] * N_x[k] + N_x[4 + j] * N_x[4 + k]) -
            wt1 * rc2 * (u[0] * N_x[j] + u[1] * N_x[4 + j]) *
                (u[0] * N_x[k] + u[1] * N_x[4 + k]);
      }
    }
  }
}}

extern "C" __global__ 
void op2_k_aero_1_res_calc_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    double *__restrict dat3,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int zero_int = 0;
    bool zero_bool = 0;
    float zero_float = 0;
    double zero_double = 0;

    double arg9_0_local[1];
    for (int d = 0; d < 1; ++d)
        arg9_0_local[d] = zero_double;

    double arg10_1_local[1];
    for (int d = 0; d < 1; ++d)
        arg10_1_local[d] = zero_double;

    double arg11_2_local[1];
    for (int d = 0; d < 1; ++d)
        arg11_2_local[d] = zero_double;

    double arg12_3_local[1];
    for (int d = 0; d < 1; ++d)
        arg12_3_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_1_res_calc::res_calc(
            dat0 + map0[0 * stride + n] * 2,
            dat0 + map0[1 * stride + n] * 2,
            dat0 + map0[2 * stride + n] * 2,
            dat0 + map0[3 * stride + n] * 2,
            dat1 + map0[0 * stride + n] * 1,
            dat1 + map0[1 * stride + n] * 1,
            dat1 + map0[2 * stride + n] * 1,
            dat1 + map0[3 * stride + n] * 1,
            dat2 + n * 16,
            arg9_0_local,
            arg10_1_local,
            arg11_2_local,
            arg12_3_local
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[0 * stride + n] * 1 + d, arg9_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[1 * stride + n] * 1 + d, arg10_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[2 * stride + n] * 1 + d, arg11_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat3 + map0[3 * stride + n] * 1 + d, arg12_3_local[d]);
    }
}

)_op2_k";


void op_par_loop_aero_1_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7,
    op_arg arg8,
    op_arg arg9,
    op_arg arg10,
    op_arg arg11,
    op_arg arg12
) {
    namespace kernel = op2_m_aero_1_res_calc;

    int n_args = 13;
    op_arg args[13];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;
    args[9] = arg9;
    args[10] = arg10;
    args[11] = arg11;
    args[12] = arg12;

    op_profile_enter_kernel("aero_1_res_calc", "c_CUDA", "Indirect (atomics)");
    op_profile_enter("Init");

    op_profile_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_aero_1_res_calc_wrapper",
                                    (void *)op2_k_aero_1_res_calc_wrapper,
                                    op2_k_aero_1_res_calc_src);

    std::array<int, 3> sections = {0, set->core_size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "aero_1_res_calc"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);

    if (first_invocation) {
        info.add_param("op2_const_m2_d", &m2, &op2_const_m2_d, &op2_const_m2_hash);
        info.add_param("op2_const_Ng2_xi_d", Ng2_xi, sizeof(op2_const_Ng2_xi_d) / sizeof(Ng2_xi[0]), op2_const_Ng2_xi_d, &op2_const_Ng2_xi_hash);
        info.add_param("op2_const_gm1_d", &gm1, &op2_const_gm1_d, &op2_const_gm1_hash);
        info.add_param("op2_const_wtg2_d", wtg2, sizeof(op2_const_wtg2_d) / sizeof(wtg2[0]), op2_const_wtg2_d, &op2_const_wtg2_hash);
        info.add_param("op2_const_gm1i_d", &gm1i, &op2_const_gm1i_d, &op2_const_gm1i_hash);

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
    arg8 = args[8];
    arg9 = args[9];
    arg10 = args[10];
    arg11 = args[11];
    arg12 = args[12];

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
                &arg4.data_d,
                &arg8.data_d,
                &arg9.data_d,
                &arg0.map_data_d,
                &start,
                &end,
                &size
            };

            void *kernel_args_jit[] = {
                &arg0.data_d,
                &arg4.data_d,
                &arg8.data_d,
                &arg9.data_d,
                &arg0.map_data_d,
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