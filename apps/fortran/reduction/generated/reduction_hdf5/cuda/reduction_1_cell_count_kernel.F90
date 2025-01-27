#define op2_s(idx, stride) 1 + ((idx) - 1) * op2_stride_##stride##_d

module op2_m_reduction_1_cell_count_main

    use cudafor
    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support
    use cudaconfigurationparams

    use op2_consts

    implicit none

    private
    public :: op2_k_reduction_1_cell_count_main

    integer(4) :: op2_stride_gbl = 0
    integer(4), constant :: op2_stride_gbl_d = 0

contains

attributes(device) &
SUBROUTINE cell_count(res, cell_count_result)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(4) :: res
  INTEGER(KIND = 4), DIMENSION(1) :: cell_count_result
  INTEGER(KIND = 4) :: d
  DO d = 1, 4
    res(d) = 0.0
  END DO
  cell_count_result = cell_count_result + 1
END SUBROUTINE

attributes(global) &
subroutine op2_k_reduction_1_cell_count_wrapper( &
    dat0, &
    gbl1, &
    start, &
    end, &
    set_size &
)
    implicit none

    ! parameters
    real(8), dimension(*) :: dat0

    integer(4), dimension(*) :: gbl1

    integer(4), value :: start, end, set_size

    ! locals
    integer(4) :: thread_id, d, n, ret

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do n = thread_id + start, end, blockDim%x * gridDim%x
        call cell_count( &
            dat0((n - 1) * 4 + 1), &
            gbl1(thread_id) &
        )
    end do
end subroutine

attributes(global) &
subroutine op2_k_reduction_1_cell_count_init_gbls( &
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

subroutine op2_k_reduction_1_cell_count_main( &
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

    integer(4), dimension(:), pointer :: gbl1
    integer(4), dimension(:), pointer, device :: gbl1_d

    real(8) :: start_time, end_time
    real(4) :: transfer

    integer(4) :: num_blocks, max_blocks, block_size, block_limit
    integer(4) :: start, end

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("reduction_1_cell_count", "CUDA", "Direct")
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

    num_blocks = (set%setptr%size + (block_size - 1)) / block_size
    num_blocks = min(num_blocks, block_limit)
    max_blocks = num_blocks

    call op_timing2_enter("Prepare GBLs")
    call prepareDeviceGbls(args, size(args), block_size * max_blocks)
    call op_timing2_exit()

    arg0 = args(1)
    arg1 = args(2)

    call c_f_pointer(arg0%data_d, dat0_d, (/4 * getsetsizefromoparg(arg0)/))

    call c_f_pointer(arg1%data, gbl1, (/1/))
    call c_f_pointer(arg1%data_d, gbl1_d, (/1 * block_size * max_blocks/))

    if (op2_stride_gbl /= block_size * max_blocks) then
        op2_stride_gbl = block_size * max_blocks
        op2_stride_gbl_d = op2_stride_gbl
    end if

    call op_timing2_enter("Init GBLs")
    call op2_k_reduction_1_cell_count_init_gbls<<<max_blocks, block_size>>>( &
        gbl1_d, &
        0 &
    )

    call op_timing2_exit()
    call op_timing2_next("Computation")
    start = 0
    end = set%setptr%size

    call op_timing2_enter("Kernel")
    call op2_k_reduction_1_cell_count_wrapper<<<num_blocks, block_size>>>( &
        dat0_d, &
        gbl1_d, &
        start, &
        end, &
        set%setptr%size &
    )


    call op_timing2_next("Process GBLs")
    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, "error in gpu kernel: ", "reduction_1_cell_count"
        print *, cudaGetErrorString(err)
    end if

    call processDeviceGbls(args, size(args), block_size * max_blocks, block_size * max_blocks)

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

module op2_m_reduction_1_cell_count_fallback

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_reduction_1_cell_count_fallback

contains

SUBROUTINE cell_count(res, cell_count_result)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(4) :: res
  INTEGER(KIND = 4), DIMENSION(1) :: cell_count_result
  INTEGER(KIND = 4) :: d
  DO d = 1, 4
    res(d) = 0.0
  END DO
  cell_count_result = cell_count_result + 1
END SUBROUTINE

subroutine op2_k_reduction_1_cell_count_wrapper( &
    dat0, &
    gbl1, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0

    integer(4), dimension(:) :: gbl1

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(2) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        call cell_count( &
            dat0(:, n), &
            gbl1(1) &
        )
    end do
end subroutine

subroutine op2_k_reduction_1_cell_count_fallback( &
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

    integer(4), pointer, dimension(:) :: gbl1

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("reduction_1_cell_count", "seq", "Direct")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))

    call c_f_pointer(arg1%data, gbl1, (/1/))

    call op2_k_reduction_1_cell_count_wrapper( &
        dat0, &
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

module op2_m_reduction_1_cell_count

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_reduction_1_cell_count_fallback
    use op2_m_reduction_1_cell_count_main

    implicit none

    private
    public :: op2_k_reduction_1_cell_count

contains

subroutine op2_k_reduction_1_cell_count( &
    name, &
    set, &
    arg0, &
    arg1 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1

    if (op_check_whitelist("reduction_1_cell_count")) then
        call op2_k_reduction_1_cell_count_main( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    else
        call op2_k_reduction_1_cell_count_fallback( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    end if
end subroutine

end module