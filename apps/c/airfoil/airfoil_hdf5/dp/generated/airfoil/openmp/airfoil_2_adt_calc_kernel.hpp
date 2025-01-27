namespace op2_k2 {
inline void adt_calc(const double *x1, const double *x2, const double *x3,
                     const double *x4, const double *q, double *adt) {
  double dx, dy, ri, u, v, c;

  ri = 1.0f / q[0];
  u = ri * q[1];
  v = ri * q[2];
  c = sqrt(gam * gm1 * (ri * q[3] - 0.5f * (u * u + v * v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  //*adt = (*adt) / cfl;
  *adt = (*adt) * (1.0f / cfl);
}
}

#define SIMD_LEN 8

void airfoil_2_adt_calc_wrapper(
    const double *__restrict__ dat0_u,
    const double *__restrict__ dat1_u,
    double *__restrict__ dat2_u,
    const int *__restrict__ map0_u,
    int map0_dim,
    int start,
    int end
) {
    const double *__restrict__ dat0 = assume_aligned(dat0_u);
    const double *__restrict__ dat1 = assume_aligned(dat1_u);
    double *__restrict__ dat2 = assume_aligned(dat2_u);
    const int *__restrict__ map0 = assume_aligned(map0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg5_local[SIMD_LEN][1];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k2::adt_calc(
                dat0 + map0[n * map0_dim + 0] * 2,
                dat0 + map0[n * map0_dim + 1] * 2,
                dat0 + map0[n * map0_dim + 2] * 2,
                dat0 + map0[n * map0_dim + 3] * 2,
                dat1 + n * 4,
                arg5_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat2 + n * 1)[d] = arg5_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k2::adt_calc(
            dat0 + map0[n * map0_dim + 0] * 2,
            dat0 + map0[n * map0_dim + 1] * 2,
            dat0 + map0[n * map0_dim + 2] * 2,
            dat0 + map0[n * map0_dim + 3] * 2,
            dat1 + n * 4,
            dat2 + n * 1
        );
    }
}

void op_par_loop_airfoil_2_adt_calc(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5
) {
    int num_args_expanded = 6;
    op_arg args_expanded[6];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;
    args_expanded[4] = arg4;
    args_expanded[5] = arg5;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): airfoil_2_adt_calc\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int num_dats_indirect = 1;
    int dats_indirect[6] = {0, 0, 0, 0, -1, -1};


#ifdef OP_PART_SIZE_2
    int part_size = OP_PART_SIZE_2;
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

            airfoil_2_adt_calc_wrapper(
                (double *)arg0.data,
                (double *)arg4.data,
                (double *)arg5.data,
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
    OP_kernels[2].time += wall_end - wall_start;

}
