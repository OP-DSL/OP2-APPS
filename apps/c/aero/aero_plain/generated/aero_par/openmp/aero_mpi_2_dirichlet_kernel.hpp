namespace op2_k2 {
inline void dirichlet(double *res) { *res = 0.0; }
}

#define SIMD_LEN 8

void aero_mpi_2_dirichlet_wrapper(
    double *__restrict__ dat0_u,
    const int *__restrict__ map0_u,
    int map0_dim,
    int start,
    int end
) {
    double *__restrict__ dat0 = assume_aligned(dat0_u);
    const int *__restrict__ map0 = assume_aligned(map0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) double arg0_0_local[SIMD_LEN][1];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k2::dirichlet(
                arg0_0_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                (dat0 + map0[n * map0_dim + 0] * 1)[d] = arg0_0_local[lane][d];
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k2::dirichlet(
            dat0 + map0[n * map0_dim + 0] * 1
        );
    }
}

void op_par_loop_aero_mpi_2_dirichlet(
    const char *name,
    op_set set,
    op_arg arg0
) {
    int num_args_expanded = 1;
    op_arg args_expanded[1];

    args_expanded[0] = arg0;

    double cpu_start, cpu_end, wall_start, wall_end;
    op_timing_realloc(2);

    OP_kernels[2].name = name;
    OP_kernels[2].count += 1;

    op_timers_core(&cpu_start, &wall_start);

    if (OP_diags > 2)
        printf(" kernel routine (indirect): aero_mpi_2_dirichlet\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int num_dats_indirect = 1;
    int dats_indirect[1] = {0};


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

            aero_mpi_2_dirichlet_wrapper(
                (double *)arg0.data,
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
