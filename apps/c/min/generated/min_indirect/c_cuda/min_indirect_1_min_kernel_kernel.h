namespace op2_m_min_indirect_1_min_kernel {


int op2_min_kernel_gbl_stride = -1;
__constant__ int op2_min_kernel_gbl_stride_d;

__device__ void min_kernel(const int *d, int *min) {
    *min = std::min(*d, *min);
}}


extern "C" __global__ 
void op2_k_min_indirect_1_min_kernel_wrapper(
    const int *__restrict dat0,
    const int *__restrict map0,
    int *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_min_indirect_1_min_kernel::min_kernel(
            dat0 + map0[0 * stride + n] * 1,
            gbl1 + thread_id
        );
    }
}


const char op2_k_min_indirect_1_min_kernel_src[] = R"_op2_k(
namespace op2_m_min_indirect_1_min_kernel {

__device__ void min_kernel(const int *d, int *min) {
    *min = std::min(*d, *min);
}}

extern "C" __global__ 
void op2_k_min_indirect_1_min_kernel_wrapper(
    const int *__restrict dat0,
    const int *__restrict map0,
    int *__restrict gbl1,
    const int start,
    const int end,
    const int stride
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = thread_id + start; i < end; i += blockDim.x * gridDim.x) {
        int n = i;



        op2_m_min_indirect_1_min_kernel::min_kernel(
            dat0 + map0[0 * stride + n] * 1,
            gbl1 + thread_id
        );
    }
}

)_op2_k";

__global__
static void op2_k_min_indirect_1_min_kernel_init_gbls(
    int *gbl1,
    int *gbl1_ref,
    int stride
) {
    namespace kernel = op2_m_min_indirect_1_min_kernel;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int d = 0; d < 1; ++d) {
        gbl1[thread_id + d * stride] = gbl1_ref[d];
    }
}

void op_par_loop_min_indirect_1_min_kernel(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    namespace kernel = op2_m_min_indirect_1_min_kernel;

    int n_args = 2;
    op_arg args[2];

    args[0] = arg0;
    args[1] = arg1;

    // op_timing2_enter_kernel("min_indirect_1_min_kernel", "c_CUDA", "Indirect (atomics)");
    // op_timing2_enter("Init");

    // op_timing2_enter("Kernel Info Setup");

    static bool first_invocation = true;
    static op::f2c::KernelInfo info("op2_k_min_indirect_1_min_kernel_wrapper",
                                    (void *)op2_k_min_indirect_1_min_kernel_wrapper,
                                    op2_k_min_indirect_1_min_kernel_src);

    std::array<int, 4> sections = {0, set->core_size, set->size, set->size + set->exec_size};

    auto [block_limit, block_size] = info.get_launch_config(nullptr, set->core_size);
    block_limit = std::min(block_limit, getBlockLimit(args, n_args, block_size, "min_indirect_1_min_kernel"));

    int max_blocks = 0;
    for (int i = 1; i < sections.size(); ++i)
        max_blocks = std::max(max_blocks, (sections[i] - sections[i - 1] + (block_size - 1)) / block_size);

    max_blocks = std::min(max_blocks, block_limit);

    if (first_invocation) {

        kernel::op2_min_kernel_gbl_stride = block_size * max_blocks;
        info.add_param("op2_min_kernel_gbl_stride_d", &kernel::op2_min_kernel_gbl_stride, &kernel::op2_min_kernel_gbl_stride_d);

        first_invocation = false;
    }

    // op_timing2_next("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges_grouped(set, n_args, args, 2);

    if (n_exec == 0) {
        // op_timing2_exit();
        // op_timing2_exit();

        op_mpi_wait_all_grouped(n_args, args, 2);

        op_mpi_reduce(&arg1, (int *)arg1.data);

        op_mpi_set_dirtybit_cuda(n_args, args);
        // op_timing2_exit();
        return;
    }

    setGblIncAtomic(false);



    static int* gbl1_ref_d = nullptr;

    // op_timing2_next("Get Kernel");
    auto *kernel_inst = info.get_kernel();
    // op_timing2_exit();


    // op_timing2_enter("Prepare GBLs");
    prepareDeviceGbls(args, n_args, block_size * max_blocks);
    bool exit_sync = false;

    arg0 = args[0];
    arg1 = args[1];

    // op_timing2_next("Update GBL Refs");
    if (gbl1_ref_d == nullptr) {
        CUDA_SAFE_CALL(cudaMalloc(&gbl1_ref_d, 1 * sizeof(int)));
    }

    CUDA_SAFE_CALL(cudaMemcpyAsync(gbl1_ref_d, arg1.data, 1 * sizeof(int), cudaMemcpyHostToDevice, 0));

    // op_timing2_next("Init GBLs");

    int stride_gbl = block_size * max_blocks;
    op2_k_min_indirect_1_min_kernel_init_gbls<<<max_blocks, block_size>>>(
        (int *)arg1.data_d,
        gbl1_ref_d,
        stride_gbl
    );

    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // op_timing2_exit();
    // op_timing2_next("Computation");

    // op_timing2_enter("Kernel");

    for (int round = 1; round < sections.size(); ++round) {
        if (round == 2) {
            // op_timing2_next("MPI Wait");
            op_mpi_wait_all_grouped(n_args, args, 2);
            // op_timing2_next("Kernel");
        }

        int start = sections[round - 1];
        int end = sections[round];

        if (end - start > 0) {
            int num_blocks = (end - start + (block_size - 1)) / block_size;
            num_blocks = std::min(num_blocks, block_limit);

            int size = f2c::round32(set->size + set->exec_size);
            void *kernel_args[] = {
                &arg0.data_d,
                &arg0.map_data_d,
                &arg1.data_d,
                &start,
                &end,
                &size
            };

            void *kernel_args_jit[] = {
                &arg0.data_d,
                &arg0.map_data_d,
                &arg1.data_d,
                &start,
                &end,
                &size
            };

            info.invoke(kernel_inst, num_blocks, block_size, kernel_args, kernel_args_jit);
        }

        if (round == 2) {
            // op_timing2_next("Process GBLs");
            exit_sync = processDeviceGbls(args, n_args, block_size * max_blocks, block_size * max_blocks);
            // op_timing2_next("Kernel");
        }
    }

    // op_timing2_exit();

    // op_timing2_exit();

    // op_timing2_enter("Finalise");
    op_mpi_reduce(&arg1, (int *)arg1.data);

    op_mpi_set_dirtybit_cuda(n_args, args);
    if (exit_sync) CUDA_SAFE_CALL(cudaStreamSynchronize(0));

    // op_timing2_exit();
    // op_timing2_exit();
}