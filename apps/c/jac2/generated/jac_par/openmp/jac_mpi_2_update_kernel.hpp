namespace op2_k2 {
inline void update(const float *r, float *du, float *u, float *u_sum,
                   float *u_max) {
  *u += *du + alpha * (*r);
  *du = 0.0f;
  *u_sum += (*u) * (*u);
  *u_max = ((*u_max > *u) ? (*u_max) : (*u));
}
}

#define SIMD_LEN 8

void jac_mpi_2_update_wrapper(
    const float *__restrict__ dat0_u,
    float *__restrict__ dat1_u,
    float *__restrict__ dat2_u,
    float *__restrict__ gbl3,
    float *__restrict__ gbl4,
    int start,
    int end
) {
    const float *__restrict__ dat0 = assume_aligned(dat0_u);
    float *__restrict__ dat1 = assume_aligned(dat1_u);
    float *__restrict__ dat2 = assume_aligned(dat2_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) float arg1_local[SIMD_LEN][3];
        alignas(SIMD_LEN * 8) float arg2_local[SIMD_LEN][2] = {0};
        alignas(SIMD_LEN * 8) float arg3_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) float arg4_local[SIMD_LEN][1];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 3; ++d) {
                arg1_local[lane][d] = (dat1 + n * 3)[d];
            }
            for (int d = 0; d < 1; ++d) {
                arg4_local[lane][d] = (gbl4)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k2::update(
                dat0 + n * 2,
                arg1_local[lane],
                arg2_local[lane],
                arg3_local[lane],
                arg4_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 3; ++d) {
                (dat1 + n * 3)[d] = arg1_local[lane][d];
            }

            for (int d = 0; d < 2; ++d) {
                (dat2 + n * 2)[d] += arg2_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl3[d] += arg3_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                gbl4[d] = MAX(gbl4[d], arg4_local[lane][d]);
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k2::update(
            dat0 + n * 2,
            dat1 + n * 3,
            dat2 + n * 2,
            gbl3,
            gbl4
        );
    }
}

void op_par_loop_jac_mpi_2_update(
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
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (direct): jac_mpi_2_update\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);


#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    float *gbl3 = (float *)arg3.data;
    float gbl3_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl3_local[thread * 64 + d] = ZERO_float;
    }

    float *gbl4 = (float *)arg4.data;
    float gbl4_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl4_local[thread * 64 + d] = gbl4[d];
    }

    #pragma omp parallel for
    for (int thread = 0; thread < num_threads; ++thread) {
        int start = (set->size * thread) / num_threads;
        int end = (set->size * (thread + 1)) / num_threads;

        jac_mpi_2_update_wrapper(
            (float *)arg0.data,
            (float *)arg1.data,
            (float *)arg2.data,
            gbl3_local + 64 * omp_get_thread_num(),
            gbl4_local + 64 * omp_get_thread_num(),
            start,
            end
        );
    }

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl3[d] += gbl3_local[thread * 64 + d];
    }
    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl4[d] = MAX(gbl4[d], gbl4_local[thread * 64 + d]);
    }

    op_mpi_reduce(&arg3, gbl3);
    op_mpi_reduce(&arg4, gbl4);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[2].time += wall_end - wall_start;

}
