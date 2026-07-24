namespace op2_m_airfoil_mpi_2_adt_calc {


__device__ inline void adt_calc(const double *x1, const double *x2, const double *x3,
                     const double *x4, const double *q, double *adt) {
  double dx, dy, ri, u, v, c;

  ri = 1.0f / q[0];
  u = ri * q[1];
  v = ri * q[2];
  c = sqrt(op2_const_gam_d * op2_const_gm1_d * (ri * q[3] - 0.5f * (u * u + v * v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  *adt = (*adt) / op2_const_cfl_d;
}}


extern "C" __global__ __launch_bounds__(128)
void op2_k_airfoil_mpi_2_adt_calc_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_airfoil_mpi_2_adt_calc::adt_calc(
            dat0 + map0[0 * stride + n] * 2,
            dat0 + map0[1 * stride + n] * 2,
            dat0 + map0[2 * stride + n] * 2,
            dat0 + map0[3 * stride + n] * 2,
            dat1 + n * 4,
            dat2 + n * 1
        );
    }
}


const char op2_k_airfoil_mpi_2_adt_calc_src[] = R"_op2_k(
namespace op2_m_airfoil_mpi_2_adt_calc {

__device__ inline void adt_calc(const double *x1, const double *x2, const double *x3,
                     const double *x4, const double *q, double *adt) {
  double dx, dy, ri, u, v, c;

  ri = 1.0f / q[0];
  u = ri * q[1];
  v = ri * q[2];
  c = sqrt(op2_const_gam_d * op2_const_gm1_d * (ri * q[3] - 0.5f * (u * u + v * v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  *adt = (*adt) / op2_const_cfl_d;
}}

extern "C" __global__ __launch_bounds__(128)
void op2_k_airfoil_mpi_2_adt_calc_wrapper(
    const double *__restrict dat0,
    const double *__restrict dat1,
    double *__restrict dat2,
    const int *__restrict map0,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_airfoil_mpi_2_adt_calc::adt_calc(
            dat0 + map0[0 * stride + n] * 2,
            dat0 + map0[1 * stride + n] * 2,
            dat0 + map0[2 * stride + n] * 2,
            dat0 + map0[3 * stride + n] * 2,
            dat1 + n * 4,
            dat2 + n * 1
        );
    }
}

)_op2_k";


void op_par_loop_airfoil_mpi_2_adt_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    namespace kernel = op2_m_airfoil_mpi_2_adt_calc;

    int n_args = 6;
    op_arg args[6];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;

    op_profile_enter_kernel("airfoil_mpi_2_adt_calc", "c_CUDA", "Indirect (atomics)");
    op_profile_enter("Init");

    op_profile_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_airfoil_mpi_2_adt_calc_wrapper",
                                    (void *)op2_k_airfoil_mpi_2_adt_calc_wrapper,
                                    op2_k_airfoil_mpi_2_adt_calc_src);

    std::array<int, 3> sections = {0, set->core_size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "airfoil_mpi_2_adt_calc"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);

    if (first_invocation) {
        info.add_param("op2_const_gam_d", &gam, &op2_const_gam_d, &op2_const_gam_hash);
        info.add_param("op2_const_cfl_d", &cfl, &op2_const_cfl_d, &op2_const_cfl_hash);
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

    op_profile_exit();

    op_profile_exit();

    op_profile_enter("Finalise");

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(hipStreamSynchronize(0));

    op_profile_exit();
    op_profile_exit();
}