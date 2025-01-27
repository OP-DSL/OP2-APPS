namespace op2_k8 {
inline void dotR(const double *r, double *c) { *c += (*r) * (*r); }
}

#define SIMD_LEN 8

void aero_8_dotR_wrapper(
    const double *__restrict__ dat0_u,
    double *__restrict__ gbl1,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][1] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k8::dotR(
                dat0 + n * 1,
                arg1_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                gbl1[d] += arg1_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k8::dotR(
            dat0 + n * 1,
            gbl1
        );
    }
}

void op_par_loop_aero_8_dotR(
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
    op_timing_realloc(8);

    OP_kernels[8].name = name;
    OP_kernels[8].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_8_dotR\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    double *gbl1 = (double *)arg1.data;
    double gbl1_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl1_local[thread * 64 + d] = ZERO_double;
    }

    #pragma omp parallel for
    for (int thread = 0; thread < num_threads; ++thread) {
        int start = (set->size * thread) / num_threads;
        int end = (set->size * (thread + 1)) / num_threads;

        aero_8_dotR_wrapper(
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
    OP_kernels[8].time += wall_end - wall_start;

}
