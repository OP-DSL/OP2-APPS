namespace op2_k5 {
inline void update(const double *qold, double *q, double *res,
                   const double *adt, double *rms) {
  double del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    *rms += del * del;
  }
}
}

#define SIMD_LEN 8

void airfoil_5_update_wrapper(
    const double *__restrict__ dat0_u,
    double *__restrict__ dat1_u,
    double *__restrict__ dat2_u,
    const double *__restrict__ dat3_u,
    double *__restrict__ gbl4,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);
    double *__restrict__ dat1 = assume_aligned(dat1_u);
    double *__restrict__ dat2 = assume_aligned(dat2_u);
    const double *__restrict__ dat3 = assume_aligned(dat3_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][4];
        alignas(SIMD_LEN * 8) double arg2_local[SIMD_LEN][4];
        alignas(SIMD_LEN * 8) double arg4_local[SIMD_LEN][1] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 4; ++d) {
                arg2_local[lane][d] = (dat2 + n * 4)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k5::update(
                dat0 + n * 4,
                arg1_local[lane],
                arg2_local[lane],
                dat3 + n * 1,
                arg4_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 4; ++d) {
                (dat1 + n * 4)[d] = arg1_local[lane][d];
            }

            for (int d = 0; d < 4; ++d) {
                (dat2 + n * 4)[d] = arg2_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl4[d] += arg4_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k5::update(
            dat0 + n * 4,
            dat1 + n * 4,
            dat2 + n * 4,
            dat3 + n * 1,
            gbl4
        );
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
    int num_args_expanded = 5;
    op_arg args_expanded[5];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(5);

    OP_kernels[5].name = name;
    OP_kernels[5].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): airfoil_5_update\n");

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

    #pragma omp parallel for
    for (int thread = 0; thread < num_threads; ++thread) {
        int start = (set->size * thread) / num_threads;
        int end = (set->size * (thread + 1)) / num_threads;

        airfoil_5_update_wrapper(
            (double *)arg0.data,
            (double *)arg1.data,
            (double *)arg2.data,
            (double *)arg3.data,
            gbl4_local + 64 * omp_get_thread_num(),
            start,
            end
        );
    }

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl4[d] += gbl4_local[thread * 64 + d];
    }

    op_mpi_reduce(&arg4, gbl4);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[5].time += wall_end - wall_start;

}
