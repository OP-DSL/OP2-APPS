namespace op2_k3 {
inline void res_calc(const float *x1, const float *x2, const float *q1,
                     const float *q2, const float *adt1, const float *adt2,
                     float *res1, float *res2) {
  float dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1 * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));
  vol1 = ri * (q1[1] * dy - q1[2] * dx);

  ri = 1.0f / q2[0];
  p2 = gm1 * (q2[3] - 0.5f * ri * (q2[1] * q2[1] + q2[2] * q2[2]));
  vol2 = ri * (q2[1] * dy - q2[2] * dx);

  mu = 0.5f * ((*adt1) + (*adt2)) * eps;

  f = 0.5f * (vol1 * q1[0] + vol2 * q2[0]) + mu * (q1[0] - q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * q2[1] + p2 * dy) +
      mu * (q1[1] - q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * q2[2] - p2 * dx) +
      mu * (q1[2] - q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (q2[3] + p2)) + mu * (q1[3] - q2[3]);
  res1[3] += f;
  res2[3] -= f;
}
}

#define SIMD_LEN 8

void airfoil_3_res_calc_wrapper(
    const float *__restrict__ dat0_u,
    const float *__restrict__ dat1_u,
    const float *__restrict__ dat2_u,
    float *__restrict__ dat3_u,
    const int *__restrict__ map0_u,
    int map0_dim,
    const int *__restrict__ map1_u,
    int map1_dim,
    int start,
    int end
) {
    const float *__restrict__ dat0 = assume_aligned(dat0_u);
    const float *__restrict__ dat1 = assume_aligned(dat1_u);
    const float *__restrict__ dat2 = assume_aligned(dat2_u);
    float *__restrict__ dat3 = assume_aligned(dat3_u);
    const int *__restrict__ map0 = assume_aligned(map0_u);
    const int *__restrict__ map1 = assume_aligned(map1_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) float arg6_0_local[SIMD_LEN][4] = {0};
        alignas(SIMD_LEN * 8) float arg7_1_local[SIMD_LEN][4] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k3::res_calc(
                dat0 + map0[n * map0_dim + 0] * 2,
                dat0 + map0[n * map0_dim + 1] * 2,
                dat1 + map1[n * map1_dim + 0] * 4,
                dat1 + map1[n * map1_dim + 1] * 4,
                dat2 + map1[n * map1_dim + 0] * 1,
                dat2 + map1[n * map1_dim + 1] * 1,
                arg6_0_local[lane],
                arg7_1_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 4; ++d) {
                (dat3 + map1[n * map1_dim + 0] * 4)[d] += arg6_0_local[lane][d];
            }

            for (int d = 0; d < 4; ++d) {
                (dat3 + map1[n * map1_dim + 1] * 4)[d] += arg7_1_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k3::res_calc(
            dat0 + map0[n * map0_dim + 0] * 2,
            dat0 + map0[n * map0_dim + 1] * 2,
            dat1 + map1[n * map1_dim + 0] * 4,
            dat1 + map1[n * map1_dim + 1] * 4,
            dat2 + map1[n * map1_dim + 0] * 1,
            dat2 + map1[n * map1_dim + 1] * 1,
            dat3 + map1[n * map1_dim + 0] * 4,
            dat3 + map1[n * map1_dim + 1] * 4
        );
    }
}

void op_par_loop_airfoil_3_res_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7
) {
    int num_args_expanded = 8;
    op_arg args_expanded[8];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;
    args_expanded[6] = arg6;
    args_expanded[7] = arg7;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(3);

    OP_kernels[3].name = name;
    OP_kernels[3].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): airfoil_3_res_calc\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int num_dats_indirect = 4;
    int dats_indirect[8] = {0, 0, 1, 1, 2, 2, 3, 3};


#ifdef OP_PART_SIZE_3
    int part_size = OP_PART_SIZE_3;
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

            airfoil_3_res_calc_wrapper(
                (float *)arg0.data,
                (float *)arg2.data,
                (float *)arg4.data,
                (float *)arg6.data,
                arg0.map_data,
                arg0.map->dim,
                arg2.map_data,
                arg2.map->dim,
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
    OP_kernels[3].time += wall_end - wall_start;

}
