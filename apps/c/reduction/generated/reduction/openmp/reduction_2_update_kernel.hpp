namespace op2_k2 {
inline void update(double *data, int *count) {
  data[0] = 0.0;
  (*count)++;
}
}

#define SIMD_LEN 8

void reduction_2_update_wrapper(
    double *__restrict__ dat0_u,
    int *__restrict__ gbl1,
    int start,
    int end
) {
    double *__restrict__ dat0 = assume_aligned(dat0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg0_local[SIMD_LEN][4];
        alignas(SIMD_LEN * 8) int arg1_local[SIMD_LEN][1] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 4; ++d) {
                arg0_local[lane][d] = (dat0 + n * 4)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k2::update(
                arg0_local[lane],
                arg1_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 4; ++d) {
                (dat0 + n * 4)[d] = arg0_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl1[d] += arg1_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k2::update(
            dat0 + n * 4,
            gbl1
        );
    }
}

void op_par_loop_reduction_2_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1
) {
    int num_args_expanded = 2;
    op_arg args_expanded[2];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): reduction_2_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    int *gbl1 = (int *)arg1.data;
    int gbl1_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl1_local[thread * 64 + d] = ZERO_int;
    }

    #pragma omp parallel for
    for (int thread = 0; thread < num_threads; ++thread) {
        int start = (set->size * thread) / num_threads;
        int end = (set->size * (thread + 1)) / num_threads;

        reduction_2_update_wrapper(
            (double *)arg0.data,
            gbl1_local + 64 * omp_get_thread_num(),
            start,
            end
        );
    }

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl1[d] += gbl1_local[thread * 64 + d];
    }

    op_mpi_reduce(&arg1, gbl1);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[2].time += wall_end - wall_start;

}
