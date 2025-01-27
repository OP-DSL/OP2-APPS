namespace op2_k3 {
inline void init_cg(const double *r, double *c, double *u, double *v, double *p) {
  *c += (*r) * (*r);
  *p = *r;
  *u = 0;
  *v = 0;
}
}

#define SIMD_LEN 8

void aero_3_init_cg_wrapper(
    const double *__restrict__ dat0_u,
    double *__restrict__ dat1_u,
    double *__restrict__ dat2_u,
    double *__restrict__ dat3_u,
    double *__restrict__ gbl1,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);
    double *__restrict__ dat1 = assume_aligned(dat1_u);
    double *__restrict__ dat2 = assume_aligned(dat2_u);
    double *__restrict__ dat3 = assume_aligned(dat3_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg2_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg3_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg4_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][1] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k3::init_cg(
                dat0 + n * 1,
                arg1_local[lane],
                arg2_local[lane],
                arg3_local[lane],
                arg4_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat1 + n * 1)[d] = arg2_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat2 + n * 1)[d] = arg3_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat3 + n * 1)[d] = arg4_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl1[d] += arg1_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k3::init_cg(
            dat0 + n * 1,
            gbl1,
            dat1 + n * 1,
            dat2 + n * 1,
            dat3 + n * 1
        );
    }
}

void op_par_loop_aero_3_init_cg(
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
    op_timing_realloc(3);

    OP_kernels[3].name = name;
    OP_kernels[3].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): aero_3_init_cg\n");

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

        aero_3_init_cg_wrapper(
            (double *)arg0.data,
            (double *)arg2.data,
            (double *)arg3.data,
            (double *)arg4.data,
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
    OP_kernels[3].time += wall_end - wall_start;

}
