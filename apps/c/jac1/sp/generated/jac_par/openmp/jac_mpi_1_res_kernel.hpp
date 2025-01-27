namespace op2_k1 {
inline void res(const float *A, const float *u, float *du, const float *beta) {
  *du += (*beta) * (*A) * (*u);
}
}

#define SIMD_LEN 8

void jac_mpi_1_res_wrapper(
    const float *__restrict__ dat0_u,
    const float *__restrict__ dat1_u,
    float *__restrict__ dat2_u,
    const int *__restrict__ map0_u,
    int map0_dim,
    const float *__restrict__ gbl3,
    int start,
    int end
) {
    const float *__restrict__ dat0 = assume_aligned(dat0_u);
    const float *__restrict__ dat1 = assume_aligned(dat1_u);
    float *__restrict__ dat2 = assume_aligned(dat2_u);
    const int *__restrict__ map0 = assume_aligned(map0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) float arg2_0_local[SIMD_LEN][1] = {0};

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k1::res(
                dat0 + n * 1,
                dat1 + map0[n * map0_dim + 1] * 1,
                arg2_0_local[lane],
                gbl3
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat2 + map0[n * map0_dim + 0] * 1)[d] += arg2_0_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k1::res(
            dat0 + n * 1,
            dat1 + map0[n * map0_dim + 1] * 1,
            dat2 + map0[n * map0_dim + 0] * 1,
            gbl3
        );
    }
}

void op_par_loop_jac_mpi_1_res(
    const char *name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3
) {
    int num_args_expanded = 4;
    op_arg args_expanded[4];

    args_expanded[0] = arg0;
    args_expanded[1] = arg1;
    args_expanded[2] = arg2;
    args_expanded[3] = arg3;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(1);

    OP_kernels[1].name = name;
    OP_kernels[1].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): jac_mpi_1_res\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int num_dats_indirect = 2;
    int dats_indirect[4] = {-1, 0, 1, -1};


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

            jac_mpi_1_res_wrapper(
                (float *)arg0.data,
                (float *)arg1.data,
                (float *)arg2.data,
                arg1.map_data,
                arg1.map->dim,
                (float *)arg3.data,
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
