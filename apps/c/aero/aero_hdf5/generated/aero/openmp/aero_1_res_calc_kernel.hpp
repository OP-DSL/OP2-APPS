namespace op2_k1 {
inline void res_calc(const double **x, const double **phim, double *K,
                     /*double *Kt,*/ double **res, double **none) {
  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < 4; k++) {
      K[j * 4 + k] = 0;
    }
  }
  for (int i = 0; i < 4; i++) { // for each gauss point
    double det_x_xi = 0;
    double N_x[8];

    double a = 0;
    for (int m = 0; m < 4; m++)
      det_x_xi += Ng2_xi[4 * i + 16 + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] = det_x_xi * Ng2_xi[4 * i + m];

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi[4 * i + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] = a * Ng2_xi[4 * i + 16 + m];

    det_x_xi *= a;

    a = 0;
    for (int m = 0; m < 4; m++)
      a += Ng2_xi[4 * i + m] * x[m][1];
    for (int m = 0; m < 4; m++)
      N_x[m] -= a * Ng2_xi[4 * i + 16 + m];

    double b = 0;
    for (int m = 0; m < 4; m++)
      b += Ng2_xi[4 * i + 16 + m] * x[m][0];
    for (int m = 0; m < 4; m++)
      N_x[4 + m] -= b * Ng2_xi[4 * i + m];

    det_x_xi -= a * b;

    for (int j = 0; j < 8; j++)
      N_x[j] /= det_x_xi;

    double wt1 = wtg2[i] * det_x_xi;
    // double wt2 = wtg2[i]*det_x_xi/r;

    double u[2] = {0.0, 0.0};
    for (int j = 0; j < 4; j++) {
      u[0] += N_x[j] * phim[j][0];
      u[1] += N_x[4 + j] * phim[j][0];
    }

    double Dk = 1.0 + 0.5 * gm1 * (m2 - (u[0] * u[0] + u[1] * u[1]));
    double rho = pow(Dk, gm1i); // wow this might be problematic -> go to log?
    double rc2 = rho / Dk;

    for (int j = 0; j < 4; j++) {
      res[j][0] += wt1 * rho * (u[0] * N_x[j] + u[1] * N_x[4 + j]);
    }
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        K[j * 4 + k] +=
            wt1 * rho * (N_x[j] * N_x[k] + N_x[4 + j] * N_x[4 + k]) -
            wt1 * rc2 * (u[0] * N_x[j] + u[1] * N_x[4 + j]) *
                (u[0] * N_x[k] + u[1] * N_x[4 + k]);
      }
    }
  }
}
}

#define SIMD_LEN 8

void aero_1_res_calc_wrapper(
    const double *__restrict__ dat0_u,
    const double *__restrict__ dat1_u,
    double *__restrict__ dat2_u,
    double *__restrict__ dat3_u,
    double *__restrict__ dat4_u,
    const int *__restrict__ map0_u,
    int map0_dim,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);
    const double *__restrict__ dat1 = assume_aligned(dat1_u);
    double *__restrict__ dat2 = assume_aligned(dat2_u);
    double *__restrict__ dat3 = assume_aligned(dat3_u);
    double *__restrict__ dat4 = assume_aligned(dat4_u);
    const int *__restrict__ map0 = assume_aligned(map0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg2_local[SIMD_LEN][16];
        alignas(SIMD_LEN * 8) double arg3_0_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg3_1_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg3_2_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg3_3_local[SIMD_LEN][1];
        alignas(SIMD_LEN * 8) double arg4_0_local[SIMD_LEN][4] = {0};
        alignas(SIMD_LEN * 8) double arg4_1_local[SIMD_LEN][4] = {0};
        alignas(SIMD_LEN * 8) double arg4_2_local[SIMD_LEN][4] = {0};
        alignas(SIMD_LEN * 8) double arg4_3_local[SIMD_LEN][4] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                arg3_0_local[lane][d] = (dat3 + map0[n * map0_dim + 0] * 1)[d];
            }

            for (int d = 0; d < 1; ++d) {
                arg3_1_local[lane][d] = (dat3 + map0[n * map0_dim + 1] * 1)[d];
            }

            for (int d = 0; d < 1; ++d) {
                arg3_2_local[lane][d] = (dat3 + map0[n * map0_dim + 2] * 1)[d];
            }

            for (int d = 0; d < 1; ++d) {
                arg3_3_local[lane][d] = (dat3 + map0[n * map0_dim + 3] * 1)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            const double *arg0_vec[] = {
                dat0 + map0[n * map0_dim + 0] * 2,
                dat0 + map0[n * map0_dim + 1] * 2,
                dat0 + map0[n * map0_dim + 2] * 2,
                dat0 + map0[n * map0_dim + 3] * 2
            };

            const double *arg1_vec[] = {
                dat1 + map0[n * map0_dim + 0] * 1,
                dat1 + map0[n * map0_dim + 1] * 1,
                dat1 + map0[n * map0_dim + 2] * 1,
                dat1 + map0[n * map0_dim + 3] * 1
            };

            double *arg3_vec[] = {
                arg3_0_local[lane],
                arg3_1_local[lane],
                arg3_2_local[lane],
                arg3_3_local[lane]
            };

            double *arg4_vec[] = {
                arg4_0_local[lane],
                arg4_1_local[lane],
                arg4_2_local[lane],
                arg4_3_local[lane]
            };

            op2_k1::res_calc(
                arg0_vec,
                arg1_vec,
                arg2_local[lane],
                arg3_vec,
                arg4_vec
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 16; ++d) {
                (dat2 + n * 16)[d] = arg2_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat3 + map0[n * map0_dim + 0] * 1)[d] = arg3_0_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat3 + map0[n * map0_dim + 1] * 1)[d] = arg3_1_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat3 + map0[n * map0_dim + 2] * 1)[d] = arg3_2_local[lane][d];
            }

            for (int d = 0; d < 1; ++d) {
                (dat3 + map0[n * map0_dim + 3] * 1)[d] = arg3_3_local[lane][d];
            }

            for (int d = 0; d < 4; ++d) {
                (dat4 + map0[n * map0_dim + 0] * 4)[d] += arg4_0_local[lane][d];
            }

            for (int d = 0; d < 4; ++d) {
                (dat4 + map0[n * map0_dim + 1] * 4)[d] += arg4_1_local[lane][d];
            }

            for (int d = 0; d < 4; ++d) {
                (dat4 + map0[n * map0_dim + 2] * 4)[d] += arg4_2_local[lane][d];
            }

            for (int d = 0; d < 4; ++d) {
                (dat4 + map0[n * map0_dim + 3] * 4)[d] += arg4_3_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        const double *arg0_vec[] = {
            dat0 + map0[n * map0_dim + 0] * 2,
            dat0 + map0[n * map0_dim + 1] * 2,
            dat0 + map0[n * map0_dim + 2] * 2,
            dat0 + map0[n * map0_dim + 3] * 2
        };

        const double *arg1_vec[] = {
            dat1 + map0[n * map0_dim + 0] * 1,
            dat1 + map0[n * map0_dim + 1] * 1,
            dat1 + map0[n * map0_dim + 2] * 1,
            dat1 + map0[n * map0_dim + 3] * 1
        };

        double *arg3_vec[] = {
            dat3 + map0[n * map0_dim + 0] * 1,
            dat3 + map0[n * map0_dim + 1] * 1,
            dat3 + map0[n * map0_dim + 2] * 1,
            dat3 + map0[n * map0_dim + 3] * 1
        };

        double *arg4_vec[] = {
            dat4 + map0[n * map0_dim + 0] * 4,
            dat4 + map0[n * map0_dim + 1] * 4,
            dat4 + map0[n * map0_dim + 2] * 4,
            dat4 + map0[n * map0_dim + 3] * 4
        };

        op2_k1::res_calc(
            arg0_vec,
            arg1_vec,
            dat2 + n * 16,
            arg3_vec,
            arg4_vec
        );
    }
}

void op_par_loop_aero_1_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4
) {
    int num_args_expanded = 17;
    op_arg args_expanded[17];

    args_expanded[0] = op_arg_dat(arg0.dat, 0, arg0.map, 2, "double", 0);
    args_expanded[1] = op_arg_dat(arg0.dat, 1, arg0.map, 2, "double", 0);
    args_expanded[2] = op_arg_dat(arg0.dat, 2, arg0.map, 2, "double", 0);
    args_expanded[3] = op_arg_dat(arg0.dat, 3, arg0.map, 2, "double", 0);
    args_expanded[4] = op_arg_dat(arg1.dat, 0, arg1.map, 1, "double", 0);
    args_expanded[5] = op_arg_dat(arg1.dat, 1, arg1.map, 1, "double", 0);
    args_expanded[6] = op_arg_dat(arg1.dat, 2, arg1.map, 1, "double", 0);
    args_expanded[7] = op_arg_dat(arg1.dat, 3, arg1.map, 1, "double", 0);
    args_expanded[8] = arg2;
    args_expanded[9] = op_opt_arg_dat(arg3.opt, arg3.dat, 0, arg3.map, 1, "double", 2);
    args_expanded[10] = op_opt_arg_dat(arg3.opt, arg3.dat, 1, arg3.map, 1, "double", 2);
    args_expanded[11] = op_opt_arg_dat(arg3.opt, arg3.dat, 2, arg3.map, 1, "double", 2);
    args_expanded[12] = op_opt_arg_dat(arg3.opt, arg3.dat, 3, arg3.map, 1, "double", 2);
    args_expanded[13] = op_opt_arg_dat(arg4.opt, arg4.dat, 0, arg4.map, 4, "double", 3);
    args_expanded[14] = op_opt_arg_dat(arg4.opt, arg4.dat, 1, arg4.map, 4, "double", 3);
    args_expanded[15] = op_opt_arg_dat(arg4.opt, arg4.dat, 2, arg4.map, 4, "double", 3);
    args_expanded[16] = op_opt_arg_dat(arg4.opt, arg4.dat, 3, arg4.map, 4, "double", 3);

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_1_res_calc\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int num_dats_indirect = 4;
    int dats_indirect[17] = {0, 0, 0, 0, 1, 1, 1, 1, -1, 2, 2, 2, 2, 3, 3, 3, 3};


#ifdef OP_PART_SIZE_1
    int part_size = OP_PART_SIZE_1;
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

            aero_1_res_calc_wrapper(
                (double *)arg0.data,
                (double *)arg1.data,
                (double *)arg2.data,
                (double *)arg3.data,
                (double *)arg4.data,
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
    OP_kernels[1].time += wall_end - wall_start;

}
