namespace op2_k7 {
inline void updateUR(double *u, double *r, const double *p, double *v,
                     const double *alpha) {
  *u += (*alpha) * (*p);
  *r -= (*alpha) * (*v);
  *v = 0.0f;
}
}

#define SIMD_LEN 8

void aero_mpi_7_updateUR_wrapper(
    double *__restrict__ dat0_u,
    double *__restrict__ dat1_u,
    const double *__restrict__ dat2_u,
    double *__restrict__ dat3_u,
    const double *__restrict__ gbl4,
    int start,
    int end
) {
    double *__restrict__ dat0 = assume_aligned(dat0_u);
    double *__restrict__ dat1 = assume_aligned(dat1_u);
    const double *__restrict__ dat2 = assume_aligned(dat2_u);
    double *__restrict__ dat3 = assume_aligned(dat3_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg0_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) double arg3_local[SIMD_LEN][1];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                arg3_local[lane][d] = (dat3 + n * 1)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k7::updateUR(
                arg0_local[lane],
                arg1_local[lane],
                dat2 + n * 1,
                arg3_local[lane],
                gbl4
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat0 + n * 1)[d] += arg0_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat1 + n * 1)[d] += arg1_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat3 + n * 1)[d] = arg3_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k7::updateUR(
            dat0 + n * 1,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1,
            gbl4
        );
    }
}

void op_par_loop_aero_mpi_7_updateUR(
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
    op_timing_realloc(7);

    OP_kernels[7].name = name;
    OP_kernels[7].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_mpi_7_updateUR\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    #pragma omp parallel for
    for (int thread = 0; thread < num_threads; ++thread) {
        int start = (set->size * thread) / num_threads;
        int end = (set->size * (thread + 1)) / num_threads;

        aero_mpi_7_updateUR_wrapper(
            (double *)arg0.data,
            (double *)arg1.data,
            (double *)arg2.data,
            (double *)arg3.data,
            (double *)arg4.data,
            start,
            end
        );
    }


    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[7].time += wall_end - wall_start;

}
