namespace op2_k1 {
inline void save_soln(const double *q, double *qold) {
  for (int n = 0; n < 4; n++)
    qold[n] = q[n];
}
}

#define SIMD_LEN 8

void airfoil_mpi_1_save_soln_wrapper(
    const double *__restrict__ dat0_u,
    double *__restrict__ dat1_u,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);
    double *__restrict__ dat1 = assume_aligned(dat1_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg1_local[SIMD_LEN][4];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k1::save_soln(
                dat0 + n * 4,
                arg1_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 4; ++d) {
                (dat1 + n * 4)[d] = arg1_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k1::save_soln(
            dat0 + n * 4,
            dat1 + n * 4
        );
    }
}

void op_par_loop_airfoil_mpi_1_save_soln(
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
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): airfoil_mpi_1_save_soln\n");

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

        airfoil_mpi_1_save_soln_wrapper(
            (double *)arg0.data,
            (double *)arg1.data,
            start,
            end
        );
    }


    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;

}
