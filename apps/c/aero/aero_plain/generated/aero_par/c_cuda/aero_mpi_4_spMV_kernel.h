namespace op2_m_aero_mpi_4_spMV {


__device__ inline void spMV(double *v0, double *v1, double *v2, double *v3, const double *K,
                 const double *p0, const double *p1, const double *p2, const double *p3) {
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += OP2_STRIDE(K, (j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += OP2_STRIDE(K, (j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = OP2_STRIDE(K, (j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v0[0] += K[0] * p0[0];
  v0[0] += K[1] * p1[0];
  v1[0] += K[1] * p0[0];
  v0[0] += K[2] * p2[0];
  v2[0] += K[2] * p0[0];
  v0[0] += K[3] * p3[0];
  v3[0] += K[3] * p0[0];
  v1[0] += K[4 + 1] * p1[0];
  v1[0] += K[4 + 2] * p2[0];
  v2[0] += K[4 + 2] * p1[0];
  v1[0] += K[4 + 3] * p3[0];
  v3[0] += K[4 + 3] * p1[0];
  v2[0] += K[8 + 2] * p2[0];
  v2[0] += K[8 + 3] * p3[0];
  v3[0] += K[8 + 3] * p2[0];
  v3[0] += K[15] * p3[0];
}}


extern "C" __global__ 
void op2_k_aero_mpi_4_spMV_wrapper(
    double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
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

    double arg0_0_local[1];
    for (int d = 0; d < 1; ++d)
        arg0_0_local[d] = zero_double;

    double arg1_1_local[1];
    for (int d = 0; d < 1; ++d)
        arg1_1_local[d] = zero_double;

    double arg2_2_local[1];
    for (int d = 0; d < 1; ++d)
        arg2_2_local[d] = zero_double;

    double arg3_3_local[1];
    for (int d = 0; d < 1; ++d)
        arg3_3_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_4_spMV::spMV(
            arg0_0_local,
            arg1_1_local,
            arg2_2_local,
            arg3_3_local,
            dat1 + n * 16,
            dat2 + map0[0 * stride + n] * 1,
            dat2 + map0[1 * stride + n] * 1,
            dat2 + map0[2 * stride + n] * 1,
            dat2 + map0[3 * stride + n] * 1
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[0 * stride + n] * 1 + d, arg0_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[1 * stride + n] * 1 + d, arg1_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[2 * stride + n] * 1 + d, arg2_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[3 * stride + n] * 1 + d, arg3_3_local[d]);
    }
}


const char op2_k_aero_mpi_4_spMV_src[] = R"_op2_k(
namespace op2_m_aero_mpi_4_spMV {

__device__ inline void spMV(double *v0, double *v1, double *v2, double *v3, const double *K,
                 const double *p0, const double *p1, const double *p2, const double *p3) {
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += OP2_STRIDE(K, (j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += OP2_STRIDE(K, (j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = OP2_STRIDE(K, (j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v0[0] += K[0] * p0[0];
  v0[0] += K[1] * p1[0];
  v1[0] += K[1] * p0[0];
  v0[0] += K[2] * p2[0];
  v2[0] += K[2] * p0[0];
  v0[0] += K[3] * p3[0];
  v3[0] += K[3] * p0[0];
  v1[0] += K[4 + 1] * p1[0];
  v1[0] += K[4 + 2] * p2[0];
  v2[0] += K[4 + 2] * p1[0];
  v1[0] += K[4 + 3] * p3[0];
  v3[0] += K[4 + 3] * p1[0];
  v2[0] += K[8 + 2] * p2[0];
  v2[0] += K[8 + 3] * p3[0];
  v3[0] += K[8 + 3] * p2[0];
  v3[0] += K[15] * p3[0];
}}

extern "C" __global__ 
void op2_k_aero_mpi_4_spMV_wrapper(
    double *__restrict dat0,
    const double *__restrict dat1,
    const double *__restrict dat2,
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

    double arg0_0_local[1];
    for (int d = 0; d < 1; ++d)
        arg0_0_local[d] = zero_double;

    double arg1_1_local[1];
    for (int d = 0; d < 1; ++d)
        arg1_1_local[d] = zero_double;

    double arg2_2_local[1];
    for (int d = 0; d < 1; ++d)
        arg2_2_local[d] = zero_double;

    double arg3_3_local[1];
    for (int d = 0; d < 1; ++d)
        arg3_3_local[d] = zero_double;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_aero_mpi_4_spMV::spMV(
            arg0_0_local,
            arg1_1_local,
            arg2_2_local,
            arg3_3_local,
            dat1 + n * 16,
            dat2 + map0[0 * stride + n] * 1,
            dat2 + map0[1 * stride + n] * 1,
            dat2 + map0[2 * stride + n] * 1,
            dat2 + map0[3 * stride + n] * 1
        );

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[0 * stride + n] * 1 + d, arg0_0_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[1 * stride + n] * 1 + d, arg1_1_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[2 * stride + n] * 1 + d, arg2_2_local[d]);

        for (int d = 0; d < 1; ++d)
            atomicAdd(dat0 + map0[3 * stride + n] * 1 + d, arg3_3_local[d]);
    }
}

)_op2_k";


void op_par_loop_aero_mpi_4_spMV(
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
    op_arg arg8
) {
    namespace kernel = op2_m_aero_mpi_4_spMV;

    int n_args = 9;
    op_arg args[9];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;

    op_profile_enter_kernel("aero_mpi_4_spMV", "c_CUDA", "Indirect (atomics)");
    op_profile_enter("Init");

    op_profile_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_aero_mpi_4_spMV_wrapper",
                                    (void *)op2_k_aero_mpi_4_spMV_wrapper,
                                    op2_k_aero_mpi_4_spMV_src);

    std::array<int, 3> sections = {0, set->core_size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "aero_mpi_4_spMV"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);

    if (first_invocation) {

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
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    op_profile_exit();
    op_profile_exit();
}