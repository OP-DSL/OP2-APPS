namespace op2_m_airfoil_5_update {


int op2_update_gbl_stride = -1;
__constant__ int op2_update_gbl_stride_d;

__device__ inline void update(const float *qold, float *q, float *res, const float *adt,
                   float *rms) {
  float del, adti;
  float rmsl = 0.0f;
  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    rmsl += del * del;
  }
  *rms += rmsl;
}}


extern "C" __global__ 
void op2_k_airfoil_5_update_wrapper(
    const float *__restrict dat0,
    float *__restrict dat1,
    float *__restrict dat2,
    const float *__restrict dat3,
    float *__restrict gbl4,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_airfoil_5_update::update(
            dat0 + n * 4,
            dat1 + n * 4,
            dat2 + n * 4,
            dat3 + n * 1,
            gbl4 + thread_id
        );
    }
}


const char op2_k_airfoil_5_update_src[] = R"_op2_k(
namespace op2_m_airfoil_5_update {

__device__ inline void update(const float *qold, float *q, float *res, const float *adt,
                   float *rms) {
  float del, adti;
  float rmsl = 0.0f;
  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    rmsl += del * del;
  }
  *rms += rmsl;
}}

extern "C" __global__ 
void op2_k_airfoil_5_update_wrapper(
    const float *__restrict dat0,
    float *__restrict dat1,
    float *__restrict dat2,
    const float *__restrict dat3,
    float *__restrict gbl4,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_airfoil_5_update::update(
            dat0 + n * 4,
            dat1 + n * 4,
            dat2 + n * 4,
            dat3 + n * 1,
            gbl4 + thread_id
        );
    }
}

)_op2_k";

__global__
static void op2_k_airfoil_5_update_init_gbls(
    float *gbl4,
    int stride
) {
    namespace kernel = op2_m_airfoil_5_update;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl4[thread_id + d * stride] = 0;
    }
}

void op_par_loop_airfoil_5_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    namespace kernel = op2_m_airfoil_5_update;

    int n_args = 5;
    op_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    op_profile_enter_kernel("airfoil_5_update", "c_CUDA", "Direct");
    op_profile_enter("Init");

    op_profile_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_airfoil_5_update_wrapper",
                                    (void *)op2_k_airfoil_5_update_wrapper,
                                    op2_k_airfoil_5_update_src);

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "airfoil_5_update"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;

    if (first_invocation) {

        kernel::op2_update_gbl_stride = block_size * max_blocks;
        info.add_param("op2_update_gbl_stride_d", &kernel::op2_update_gbl_stride, &kernel::op2_update_gbl_stride_d);

        first_invocation = false;
    }

    op_profile_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        op_profile_exit();
        op_profile_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg4, (float *)arg4.data);

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

    op_profile_next("Update GBL Refs");

    op_profile_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_airfoil_5_update_init_gbls<<<max_blocks, block_size>>>(
        (float *)arg4.data_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(cudaPeekAtLastError());

    op_profile_exit();
    op_profile_next("Computation");

    int start = 0;
    int end = set->size;

    op_profile_enter("Kernel");

    int size = f2c::round32(set->size);
    void *kernel_args[] = {
        &arg0.data_d,
        &arg1.data_d,
        &arg2.data_d,
        &arg3.data_d,
        &arg4.data_d,
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
        &start,
        &end,
        &size
    };

    info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);

    op_profile_next("Process GBLs");
    exit_sync = processDeviceGbls(args, n_args, block_size * max_blocks, block_size * max_blocks);

    op_profile_exit();

    op_profile_exit();

    op_profile_enter("Finalise");
    op_mpi_reduce(&arg4, (float *)arg4.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    op_profile_exit();
    op_profile_exit();
}