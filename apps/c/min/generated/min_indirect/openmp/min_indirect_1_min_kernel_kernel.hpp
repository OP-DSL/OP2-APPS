namespace op2_k1 {
void min_kernel(const int *d, int *min) {
    *min = std::min(*d, *min);
}
}

#define SIMD_LEN 8

void min_indirect_1_min_kernel_wrapper(
    const int *__restrict__ dat0_u,
    const int *__restrict__ map0_u,
    int map0_dim,
    int *__restrict__ gbl1,
    int start,
    int end
) {
    const int *__restrict__ dat0 = assume_aligned(dat0_u);
    const int *__restrict__ map0 = assume_aligned(map0_u);

    int block = start;
    for (; block + SIMD_LEN < end; block += SIMD_LEN) {
        alignas(SIMD_LEN * 8) int arg1_local[SIMD_LEN][1];

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                arg1_local[lane][d] = (gbl1)[d];
            }
        }

        #pragma omp simd
        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            op2_k1::min_kernel(
                dat0 + map0[n * map0_dim + 0] * 1,
                arg1_local[lane]
            );
        }

        for (int lane = 0; lane < SIMD_LEN; ++lane) {
            int n = block + lane;

            for (int d = 0; d < 1; ++d) {
                gbl1[d] = MIN(gbl1[d], arg1_local[lane][d]);
            }
        }
    }

    for (int n = block; n < end; ++n) {
        op2_k1::min_kernel(
            dat0 + map0[n * map0_dim + 0] * 1,
            gbl1
        );
    }
}

void op_par_loop_min_indirect_1_min_kernel(
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
        printf(" kernel routine (indirect): min_indirect_1_min_kernel\n");

    int set_size = op_mpi_halo_exchanges(set, num_args_expanded, args_expanded);

    int num_dats_indirect = 1;
    int dats_indirect[2] = {0, -1};


#ifdef OP_PART_SIZE_1
    int part_size = OP_PART_SIZE_1;
#else
    int part_size = OP_part_size;
#endif

    op_plan *plan = op_plan_get_stage_upload(name, set, part_size, num_args_expanded, args_expanded,
        num_dats_indirect, dats_indirect, OP_STAGE_ALL, 0);

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
#else
    int num_threads = 1;
#endif

    int *gbl1 = (int *)arg1.data;
    int gbl1_local[num_threads * 64];

    for (int thread = 0; thread < num_threads; ++thread) {
        for (int d = 0; d < 1; ++d)
            gbl1_local[thread * 64 + d] = gbl1[d];
    }

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

            min_indirect_1_min_kernel_wrapper(
                (int *)arg0.data,
                arg0.map_data,
                arg0.map->dim,
                gbl1_local + 64 * omp_get_thread_num(),
                offset,
                offset + num_elem
            );
        }

        block_offset += num_blocks;

        if (col != plan->ncolors_owned - 1)
            continue;

        for (int thread = 0; thread < num_threads; ++thread) {
            for (int d = 0; d < 1; ++d)
                gbl1[d] = MIN(gbl1[d], gbl1_local[thread * 64 + d]);
        }
    }

    if (set_size == set->core_size)
        op_mpi_wait_all(num_args_expanded, args_expanded);

    op_mpi_reduce(&arg1, gbl1);
    op_mpi_set_dirtybit(num_args_expanded, args_expanded);

    op_timers_core(&cpu_end, &wall_end);
    OP_kernels[1].time += wall_end - wall_start;

}
