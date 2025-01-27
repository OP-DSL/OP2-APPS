namespace op2_k10 {
inline void update(double *phim, double *res, const double *u, double *rms) {
  *phim -= *u;
  *res = 0.0;
  *rms += (*u) * (*u);
}
}

#define SIMD_LEN 8

void aero_mpi_10_update_wrapper(
    double *__restrict__ dat0_u,
    double *__restrict__ dat1_u,
    const double *__restrict__ dat2_u,
    double *__restrict__ gbl3,
    int start,
    int end
) {
    double *__restrict__ dat0 = assume_aligned(dat0_u);
    double *__restrict__ dat1 = assume_aligned(dat1_u);
    const double *__restrict__ dat2 = assume_aligned(dat2_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg0_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg3_local[SIMD_LEN][1] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                arg0_local[lane][d] = (dat0 + n * 1)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k10::update(
                arg0_local[lane],
                arg1_local[lane],
                dat2 + n * 1,
                arg3_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat0 + n * 1)[d] = arg0_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat1 + n * 1)[d] = arg1_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl3[d] += arg3_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k10::update(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            gbl3
        );
    }
}

void op_par_loop_aero_mpi_10_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    int num_args_expanded = 4;
    op_arg args_expanded[4];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(10);

    OP_kernels[10].name = name;
    OP_kernels[10].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_10_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    double *gbl3 = (double *)arg3.data;
    double gbl3_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl3_local[thread * 64 + d] = ZERO_double;
    }

    #pragma omp parallel for
    for (int thread = 0; thread < num_threads; ++thread) {
        int start = (set->size * thread) / num_threads;
        int end = (set->size * (thread + 1)) / num_threads;

        aero_mpi_10_update_wrapper(
            (double *)arg0.data,
            (double *)arg1.data,
            (double *)arg2.data,
            gbl3_local + 64 * omp_get_thread_num(),
            start,
            end
        );
    }

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl3[d] += gbl3_local[thread * 64 + d];
    }

    op_mpi_reduce(&arg3, gbl3);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[10].time += wall_end - wall_start;

}
