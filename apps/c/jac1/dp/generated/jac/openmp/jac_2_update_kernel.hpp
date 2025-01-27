namespace op2_k2 {
static inline double maxfun(double a, double b) {
   return a>b ? a : b;
}

inline void update(const double *r, double *du, double *u, int *index, double *u_sum,
                   double *u_max) {
  *u += *du + alpha * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = maxfun(*u_max, *u);
}
}

#define SIMD_LEN 8

void jac_2_update_wrapper(
    const double *__restrict__ dat0_u,
    double *__restrict__ dat1_u,
    double *__restrict__ dat2_u,
    double *__restrict__ gbl4,
    double *__restrict__ gbl5,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);
    double *__restrict__ dat1 = assume_aligned(dat1_u);
    double *__restrict__ dat2 = assume_aligned(dat2_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg2_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) double arg4_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) double arg5_local[SIMD_LEN][1];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                arg1_local[lane][d] = (dat1 + n * 1)[d];
            }
            for (int d = 0; d < 1; ++d) {
                arg5_local[lane][d] = (gbl5)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k2::update(
                dat0 + n * 1,
                arg1_local[lane],
                arg2_local[lane],
                arg3_local[lane],
                arg4_local[lane],
                arg5_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat1 + n * 1)[d] = arg1_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat2 + n * 1)[d] += arg2_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl4[d] += arg4_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl5[d] = MAX(gbl5[d], arg5_local[lane][d]);
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k2::update(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            dat + n * ,
            gbl4,
            gbl5
        );
    }
}

void op_par_loop_jac_2_update(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    int num_args_expanded = 5;
    op_arg args_expanded[5];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg4;
    args_expanded[4] = arg5;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): jac_2_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    double *gbl4 = (double *)arg4.data;
    double gbl4_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl4_local[thread * 64 + d] = ZERO_double;
    }

    double *gbl5 = (double *)arg5.data;
    double gbl5_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl5_local[thread * 64 + d] = gbl5[d];
    }

    #pragma omp parallel for
    for (int thread = 0; thread < num_threads; ++thread) {
        int start = (set->size * thread) / num_threads;
        int end = (set->size * (thread + 1)) / num_threads;

        jac_2_update_wrapper(
            (double *)arg0.data,
            (double *)arg1.data,
            (double *)arg2.data,
            gbl4_local + 64 * omp_get_thread_num(),
            gbl5_local + 64 * omp_get_thread_num(),
            start,
            end
        );
    }

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl4[d] += gbl4_local[thread * 64 + d];
    }
    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl5[d] = MAX(gbl5[d], gbl5_local[thread * 64 + d]);
    }

    op_mpi_reduce(&arg4, gbl4);
    op_mpi_reduce(&arg5, gbl5);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[2].time += wall_end - wall_start;

}
