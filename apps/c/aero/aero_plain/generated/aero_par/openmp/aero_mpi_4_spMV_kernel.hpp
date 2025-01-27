namespace op2_k4 {
inline void spMV(double **v, const double *K, const double **p) {
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += OP2_STRIDE(K, (j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += OP2_STRIDE(K, (j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = OP2_STRIDE(K, (j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v[0][0] += K[0] * p[0][0];
  v[0][0] += K[1] * p[1][0];
  v[1][0] += K[1] * p[0][0];
  v[0][0] += K[2] * p[2][0];
  v[2][0] += K[2] * p[0][0];
  v[0][0] += K[3] * p[3][0];
  v[3][0] += K[3] * p[0][0];
  v[1][0] += K[4 + 1] * p[1][0];
  v[1][0] += K[4 + 2] * p[2][0];
  v[2][0] += K[4 + 2] * p[1][0];
  v[1][0] += K[4 + 3] * p[3][0];
  v[3][0] += K[4 + 3] * p[1][0];
  v[2][0] += K[8 + 2] * p[2][0];
  v[2][0] += K[8 + 3] * p[3][0];
  v[3][0] += K[8 + 3] * p[2][0];
  v[3][0] += K[15] * p[3][0];
}
}

#define SIMD_LEN 8

void aero_mpi_4_spMV_wrapper(
    double *__restrict__ dat0_u,
    const double *__restrict__ dat1_u,
    const double *__restrict__ dat2_u,
    const int *__restrict__ map0_u,
    int map0_dim,
    int start,
    int end
) {
    double *__restrict__ dat0 = assume_aligned(dat0_u);
    const double *__restrict__ dat1 = assume_aligned(dat1_u);
    const double *__restrict__ dat2 = assume_aligned(dat2_u);
    const int *__restrict__ map0 = assume_aligned(map0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg0_0_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) double arg0_1_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) double arg0_2_local[SIMD_LEN][1] = {0};
        alignas(SIMD_LEN * 8) double arg0_3_local[SIMD_LEN][1] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            double *arg0_vec[] = {
                arg0_0_local[lane],
                arg0_1_local[lane],
                arg0_2_local[lane],
                arg0_3_local[lane]
            };

            const double *arg2_vec[] = {
                dat2 + map0[n * map0_dim + 0] * 1,
                dat2 + map0[n * map0_dim + 1] * 1,
                dat2 + map0[n * map0_dim + 2] * 1,
                dat2 + map0[n * map0_dim + 3] * 1
            };

            op2_k4::spMV(
                arg0_vec,
                dat1 + n * 16,
                arg2_vec
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat0 + map0[n * map0_dim + 0] * 1)[d] += arg0_0_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat0 + map0[n * map0_dim + 1] * 1)[d] += arg0_1_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat0 + map0[n * map0_dim + 2] * 1)[d] += arg0_2_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat0 + map0[n * map0_dim + 3] * 1)[d] += arg0_3_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        double *arg0_vec[] = {
            dat0 + map0[n * map0_dim + 0] * 1,
            dat0 + map0[n * map0_dim + 1] * 1,
            dat0 + map0[n * map0_dim + 2] * 1,
            dat0 + map0[n * map0_dim + 3] * 1
        };

        const double *arg2_vec[] = {
            dat2 + map0[n * map0_dim + 0] * 1,
            dat2 + map0[n * map0_dim + 1] * 1,
            dat2 + map0[n * map0_dim + 2] * 1,
            dat2 + map0[n * map0_dim + 3] * 1
        };

        op2_k4::spMV(
            arg0_vec,
            dat1 + n * 16,
            arg2_vec
        );
    }
}

void op_par_loop_aero_mpi_4_spMV(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int num_args_expanded = 9;
    op_arg args_expanded[9];

    args_expanded[0] = op_arg_dat(arg0.dat, 0, arg0.map, 1, "double", 3);
    args_expanded[1] = op_arg_dat(arg0.dat, 1, arg0.map, 1, "double", 3);
    args_expanded[2] = op_arg_dat(arg0.dat, 2, arg0.map, 1, "double", 3);
    args_expanded[3] = op_arg_dat(arg0.dat, 3, arg0.map, 1, "double", 3);
    args_expanded[4] = arg1;
    args_expanded[5] = op_arg_dat(arg2.dat, 0, arg2.map, 1, "double", 0);
    args_expanded[6] = op_arg_dat(arg2.dat, 1, arg2.map, 1, "double", 0);
    args_expanded[7] = op_arg_dat(arg2.dat, 2, arg2.map, 1, "double", 0);
    args_expanded[8] = op_arg_dat(arg2.dat, 3, arg2.map, 1, "double", 0);

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(4);

    OP_kernels[4].name = name;
    OP_kernels[4].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_mpi_4_spMV\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int num_dats_indirect = 2;
    int dats_indirect[9] = {0, 0, 0, 0, -1, 1, 1, 1, 1};


#ifdef OP_PART_SIZE_4
    int part_size = OP_PART_SIZE_4;
#else
    int part_size = OP_part_size;
#endif

    op_plan *plan = op_plan_get_stage_upload(name, set, part_size, num_args_expanded, args_expanded,
        num_dats_indirect, dats_indirect, OP_STAGE_ALL, 0);

    int block_offset = 0;
    for (int col = 0; col < plan->ncolors; ++col) {
        if (col == plan->ncolors_core)
            op_mpi_wait_all(num_args_expanded, args_expanded);

        int num_blocks = plan->ncolblk[col];

        #pragma omp parallel for
        for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
            int block_id = plan->blkmap[block_idx + block_offset];
            int num_elem = plan->nelems[block_id];
            int offset = plan->offset[block_id];

            aero_mpi_4_spMV_wrapper(
                (double *)arg0.data,
                (double *)arg1.data,
                (double *)arg2.data,
                arg0.map_data,
                arg0.map->dim,
                offset,
                offset + num_elem
            );
        }

        block_offset += num_blocks;
    }

    if (set_size == set->core_size)
        op_mpi_wait_all(num_args_expanded, args_expanded);

    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[4].time += wall_end - wall_start;

}
