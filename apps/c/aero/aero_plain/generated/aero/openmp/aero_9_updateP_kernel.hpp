namespace op2_k9 {
inline void updateP(const double *r, double *p, const double *beta) {
  *p = (*beta) * (*p) + (*r);
}
}

#define SIMD_LEN 8

void aero_9_updateP_wrapper(
    const double *__restrict__ dat0_u,
    double *__restrict__ dat1_u,
    const double *__restrict__ gbl2,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);
    double *__restrict__ dat1 = assume_aligned(dat1_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][1];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                arg1_local[lane][d] = (dat1 + n * 1)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k9::updateP(
                dat0 + n * 1,
                arg1_local[lane],
                gbl2
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat1 + n * 1)[d] = arg1_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k9::updateP(
            dat0 + n * 1,
            dat1 + n * 1,
            gbl2
        );
    }
}

void op_par_loop_aero_9_updateP(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int num_args_expanded = 3;
    op_arg args_expanded[3];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(9);

    OP_kernels[9].name = name;
    OP_kernels[9].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_9_updateP\n");

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

        aero_9_updateP_wrapper(
            (double *)arg0.data,
            (double *)arg1.data,
            (double *)arg2.data,
            start,
            end
        );
    }


    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[9].time += wall_end - wall_start;

}
