namespace op2_m_jac_mpi_2_update {


int op2_update_gbl_stride = -1;
__constant__ int op2_update_gbl_stride_d;

__device__ inline void update(const float *r, float *du, float *u, float *u_sum,
                   float *u_max) {
  *u += *du + op2_const_alpha_d * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = ((*u_max > *u) ? (*u_max) : (*u));
}}


extern "C" __global__ 
void op2_k_jac_mpi_2_update_wrapper(
    const float *__restrict dat0,
    float *__restrict dat1,
    float *__restrict dat2,
    float *__restrict gbl3,
    float *__restrict gbl4,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_jac_mpi_2_update::update(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            gbl3 + thread_id,
            gbl4 + thread_id
        );
    }
}


const char op2_k_jac_mpi_2_update_src[] = R"_op2_k(
namespace op2_m_jac_mpi_2_update {

__device__ inline void update(const float *r, float *du, float *u, float *u_sum,
                   float *u_max) {
  *u += *du + op2_const_alpha_d * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = ((*u_max > *u) ? (*u_max) : (*u));
}}

extern "C" __global__ 
void op2_k_jac_mpi_2_update_wrapper(
    const float *__restrict dat0,
    float *__restrict dat1,
    float *__restrict dat2,
    float *__restrict gbl3,
    float *__restrict gbl4,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_jac_mpi_2_update::update(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            gbl3 + thread_id,
            gbl4 + thread_id
        );
    }
}

)_op2_k";

__global__
static void op2_k_jac_mpi_2_update_init_gbls(
    float *gbl3,
    float *gbl4,
    float *gbl4_ref,
    int stride
) {
    namespace kernel = op2_m_jac_mpi_2_update;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl3[thread_id + d * stride] = 0;
    }
    for (int d = 0; d < 1; ++d) {
        gbl4[thread_id + d * stride] = gbl4_ref[d];
    }
}

void op_par_loop_jac_mpi_2_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    namespace kernel = op2_m_jac_mpi_2_update;

    int n_args = 5;
    op_arg args[5];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;

    // op_timing2_enter_kernel("jac_mpi_2_update", "c_CUDA", "Direct");
    // op_timing2_enter("Init");

    // op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_jac_mpi_2_update_wrapper",
                                    (void *)op2_k_jac_mpi_2_update_wrapper,
                                    op2_k_jac_mpi_2_update_src);

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "jac_mpi_2_update"));

    int num_blocks = (set->size + (block_size - 1)) / block_size;
    num_blocks = std::min(num_blocks, block_limit);
    int max_blocks = num_blocks;

    if (first_invocation) {
        info.add_param("op2_const_alpha_d", &alpha, &op2_const_alpha_d, &op2_const_alpha_hash);

        kernel::op2_update_gbl_stride = block_size * max_blocks;
        info.add_param("op2_update_gbl_stride_d", &kernel::op2_update_gbl_stride, &kernel::op2_update_gbl_stride_d);

        first_invocation = false;
    }

    // op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        // op_timing2_exit();
        // op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg3, (float *)arg3.data);
        op_mpi_reduce(&arg4, (float *)arg4.data);

        op_mpi_set_dirtybit_cuda(n_args, args);
        // op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);



    static float* gbl4_ref_d = nullptr;

    // op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    // op_timing2_exit();


    // op_timing2_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];
    arg2 = args[2];
    arg3 = args[3];
    arg4 = args[4];

    // op_timing2_next("Update GBL Refs");
    if (gbl4_ref_d == nullptr) {
        CUDA_SAFE_CALL(cudaMalloc(&gbl4_ref_d, 1 * sizeof(float)));
    }

    CUDA_SAFE_CALL(cudaMemcpyAsync(gbl4_ref_d, arg4.data, 1 * sizeof(float), cudaMemcpyHostToDevice, 0));

    // op_timing2_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_jac_mpi_2_update_init_gbls<<<max_blocks, block_size>>>(
        (float *)arg3.data_d,
        (float *)arg4.data_d,
        gbl4_ref_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // op_timing2_exit();
    // op_timing2_next("Computation");

    int start = 0;
    int end = set->size;

    // op_timing2_enter("Kernel");

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

    // op_timing2_next("Process GBLs");
    exit_sync = processDeviceGbls(args, n_args, block_size * max_blocks, block_size * max_blocks);

    // op_timing2_exit();

    // op_timing2_exit();

    // op_timing2_enter("Finalise");
    op_mpi_reduce(&arg3, (float *)arg3.data);
    op_mpi_reduce(&arg4, (float *)arg4.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    // op_timing2_exit();
    // op_timing2_exit();
}