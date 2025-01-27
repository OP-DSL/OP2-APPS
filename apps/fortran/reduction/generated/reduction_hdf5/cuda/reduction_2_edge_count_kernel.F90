#define op2_s(idx, stride) 1 + ((idx) - 1) * op2_stride_##stride##_d

module op2_m_reduction_2_edge_count_main

    use cudafor
    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support
    use cudaconfigurationparams

    use op2_consts

    implicit none

    private
    public :: op2_k_reduction_2_edge_count_main

    integer(4) :: op2_stride_gbl = 0
    integer(4), constant :: op2_stride_gbl_d = 0

contains

attributes(device) &
SUBROUTINE edge_count(res, edge_count_result)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(4) :: res
  INTEGER(KIND = 4), DIMENSION(1) :: edge_count_result
  INTEGER(KIND = 4) :: d
  DO d = 1, 4
    res(d) = 0.0
  END DO
  edge_count_result = edge_count_result + 1
END SUBROUTINE

attributes(global) &
subroutine op2_k_reduction_2_edge_count_wrapper( &
    dat0, &
    map0, &
    gbl1, &
    start, &
    end, &
    col_reord, &
    set_size &
)
    implicit none

    ! parameters
    real(8), dimension(*) :: dat0

    integer(4), dimension(*) :: map0

    integer(4), dimension(*) :: gbl1

    integer(4), value :: start, end, set_size
    integer(4), dimension(*) :: col_reord

    ! locals
    integer(4) :: thread_id, d, n, ret, m

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do m = thread_id + start, end, blockDim%x * gridDim%x
        n = col_reord(m) + 1

        call edge_count( &
            dat0(map0(0 * set_size + n) * 4 + 1), &
            gbl1(thread_id) &
        )
    end do
end subroutine

attributes(global) &
subroutine op2_k_reduction_2_edge_count_init_gbls( &
    gbl1, &
    dummy &
)
    implicit none

    ! parameters
    integer(4), dimension(*) :: gbl1

    integer(4), value :: dummy

    ! locals
    integer(4) :: thread_id, d

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do d = 1, 1
        gbl1(thread_id + (d - 1) * op2_stride_gbl_d) = 0
    end do
end subroutine

subroutine op2_k_reduction_2_edge_count_main( &
    name, &
    set, &
    arg0, &
    arg1 &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1

    ! locals
    type(op_arg), dimension(2) :: args

    integer(4) :: n_exec, col, block, round, dim, err, d

    real(8), dimension(:), pointer, device :: dat0_d

    integer(4), dimension(:, :), pointer, device :: map0_d

    integer(4), dimension(:), pointer :: gbl1
    integer(4), dimension(:), pointer, device :: gbl1_d

    real(8) :: start_time, end_time
    real(4) :: transfer

    integer(4) :: num_blocks, max_blocks, block_size, block_limit
    integer(4) :: start, end

    integer(4) :: num_dats_indirect
    integer(4), dimension(2) :: dats_indirect

    integer(4) :: part_size

    type(op_plan), pointer :: plan
    integer(4), dimension(:), pointer :: plan_ncolblk, plan_color2_offsets
    integer(4), dimension(:), pointer, device :: plan_col_reord

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("reduction_2_edge_count", "CUDA", "Indirect (colouring)")
    call op_timing2_enter("Init")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges_grouped(set%setcptr, size(args), args, 2)

    if (n_exec == 0) then
        call op_timing2_exit()
        call op_timing2_exit()

        call op_mpi_wait_all_grouped(size(args), args, 2)
        call op_mpi_reduce_int(arg1, arg1%data)
        call op_mpi_set_dirtybit_cuda(size(args), args)
        err = cudaDeviceSynchronize()

        if (err /= 0) then
            print *, cudaGetErrorString(err)
        end if

        call op_timing2_exit()
        return
    end if

    call op_timing2_next("Update consts")
    call op_timing2_exit()

    call setGblIncAtomic(logical(.false., c_bool))
    block_size = getBlockSize(name // c_null_char, set%setptr%size)
    block_limit = getBlockLimit(args, size(args), block_size, name // c_null_char)

    num_dats_indirect = 1
    dats_indirect = (/0, -1/)

    call op_timing2_enter("Plan")

    part_size = getpartitionsize(name // c_null_char, set%setptr%size)
    plan => fortranplancaller( &
        name // c_null_char, &
        set%setcptr, &
        part_size, &
        size(args), &
        args, &
        num_dats_indirect, &
        dats_indirect, &
        4 &
    )

    call c_f_pointer(plan%ncolblk, plan_ncolblk, (/ plan%ncolors /))
    call c_f_pointer(plan%color2_offsets, plan_color2_offsets, (/ plan%ncolors + 1 /))
    call c_f_pointer(plan%col_reord, plan_col_reord, (/ set%setptr%size + set%setptr%exec_size /))

    max_blocks = 0
    do col = 1, plan%ncolors
        start = plan_color2_offsets(col)
        end = plan_color2_offsets(col + 1)

        num_blocks = (end - start + (block_size - 1)) / block_size
        num_blocks = min(num_blocks, block_limit)
        max_blocks = max(max_blocks, num_blocks)
    end do

    call op_timing2_exit()

    call op_timing2_enter("Prepare GBLs")
    call prepareDeviceGbls(args, size(args), block_size * max_blocks)
    call op_timing2_exit()

    arg0 = args(1)
    arg1 = args(2)

    call c_f_pointer(arg0%data_d, dat0_d, (/4 * getsetsizefromoparg(arg0)/))

    call c_f_pointer(arg0%map_data_d, map0_d, (/set%setptr%size, getmapdimfromoparg(arg0)/))

    call c_f_pointer(arg1%data, gbl1, (/1/))
    call c_f_pointer(arg1%data_d, gbl1_d, (/1 * block_size * max_blocks/))

    if (op2_stride_gbl /= block_size * max_blocks) then
        op2_stride_gbl = block_size * max_blocks
        op2_stride_gbl_d = op2_stride_gbl
    end if

    call op_timing2_enter("Init GBLs")
    call op2_k_reduction_2_edge_count_init_gbls<<<max_blocks, block_size>>>( &
        gbl1_d, &
        0 &
    )

    call op_timing2_exit()
    call op_timing2_next("Computation")
    call op_timing2_enter("Kernel")
    do col = 1, plan%ncolors
        if (col == plan%ncolors_core + 1) then
            call op_timing2_next("MPI Wait")
            call op_mpi_wait_all_grouped(size(args), args, 2)
            call op_timing2_next("Kernel")
        end if

        start = plan_color2_offsets(col)
        end = plan_color2_offsets(col + 1)

        num_blocks = (end - start + (block_size - 1)) / block_size
        num_blocks = min(num_blocks, block_limit)

        call op2_k_reduction_2_edge_count_wrapper<<<num_blocks, block_size>>>( &
            dat0_d, &
            map0_d, &
            gbl1_d, &
            start, &
            end, &
            plan_col_reord, &
            set%setptr%size + set%setptr%exec_size &
        )

        if (col == plan%ncolors_owned) then
            call op_timing2_next("Process GBLs")
            err = cudaDeviceSynchronize()

            if (err /= 0) then
                print *, "error in gpu kernel: ", "reduction_2_edge_count"
                print *, cudaGetErrorString(err)
            end if

            call processDeviceGbls(args, size(args), block_size * max_blocks, block_size * max_blocks)
            call op_timing2_next("Kernel")
        end if
    end do

    call op_timing2_exit()
    call op_timing2_exit()

    call op_timing2_enter("Finalise")
    call op_mpi_reduce_int(arg1, arg1%data)
    call op_mpi_set_dirtybit_cuda(size(args), args)

    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, cudaGetErrorString(err)
    end if

    call op_timing2_exit()
    call op_timing2_exit()
end subroutine

end module

module op2_m_reduction_2_edge_count_fallback

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_reduction_2_edge_count_fallback

contains

SUBROUTINE edge_count(res, edge_count_result)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(4) :: res
  INTEGER(KIND = 4), DIMENSION(1) :: edge_count_result
  INTEGER(KIND = 4) :: d
  DO d = 1, 4
    res(d) = 0.0
  END DO
  edge_count_result = edge_count_result + 1
END SUBROUTINE

subroutine op2_k_reduction_2_edge_count_wrapper( &
    dat0, &
    map0, &
    gbl1, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0

    integer(4), dimension(:, :) :: map0

    integer(4), dimension(:) :: gbl1

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(2) :: args

    ! locals
    integer(4), dimension(size(gbl1)) :: gbl1_temp

    integer(4) :: n

    gbl1_temp = gbl1

    do n = 1, n_exec
        if (n == set%setptr%core_size + 1) then
            call op_timing2_next("MPI Wait")
            call op_mpi_wait_all(size(args), args)
            call op_timing2_next("Computation")
        end if

        if (n == set%setptr%size + 1) then
            gbl1 = gbl1_temp
        end if

        call edge_count( &
            dat0(:, map0(1, n) + 1), &
            gbl1_temp(1) &
        )
    end do

    if (n_exec <= set%setptr%size) then
        gbl1 = gbl1_temp
    end if
end subroutine

subroutine op2_k_reduction_2_edge_count_fallback( &
    name, &
    set, &
    arg0, &
    arg1 &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1

    ! locals
    type(op_arg), dimension(2) :: args

    integer(4) :: n_exec

    real(8), pointer, dimension(:, :) :: dat0

    integer(4), pointer, dimension(:, :) :: map0

    integer(4), pointer, dimension(:) :: gbl1

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("reduction_2_edge_count", "seq", "Indirect")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))

    call c_f_pointer(arg0%map_data, map0, (/getmapdimfromoparg(arg0), set%setptr%size/))

    call c_f_pointer(arg1%data, gbl1, (/1/))

    call op2_k_reduction_2_edge_count_wrapper( &
        dat0, &
        map0, &
        gbl1, &
        n_exec, &
        set, &
        args &
    )

    call op_timing2_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_timing2_next("MPI Reduce")

    call op_mpi_reduce_int(arg1, arg1%data)

    call op_timing2_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_timing2_exit()
end subroutine

end module

module op2_m_reduction_2_edge_count

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_reduction_2_edge_count_fallback
    use op2_m_reduction_2_edge_count_main

    implicit none

    private
    public :: op2_k_reduction_2_edge_count

contains

subroutine op2_k_reduction_2_edge_count( &
    name, &
    set, &
    arg0, &
    arg1 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1

    if (op_check_whitelist("reduction_2_edge_count")) then
        call op2_k_reduction_2_edge_count_main( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    else
        call op2_k_reduction_2_edge_count_fallback( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    end if
end subroutine

end module