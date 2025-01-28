#define op2_s(idx, stride) 1 + ((idx) - 1) * op2_stride_##stride##_d

module op2_m_airfoil_1_save_soln_main

    use cudafor
    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support
    use cudaconfigurationparams

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_1_save_soln_main

contains

attributes(device) &
SUBROUTINE save_soln(q, qold)
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
  REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: qold
  INTEGER(KIND = 4) :: i
  DO i = 1, 4
    qold(i) = q(i)
  END DO
END SUBROUTINE

attributes(global) &
subroutine op2_k_airfoil_1_save_soln_wrapper( &
    dat0, &
    dat1, &
    start, &
    end, &
    set_size &
)
    implicit none

    ! parameters
    real(8), dimension(*) :: dat0
    real(8), dimension(*) :: dat1

    integer(4), value :: start, end, set_size

    ! locals
    integer(4) :: thread_id, d, n, ret

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do n = thread_id + start, end, blockDim%x * gridDim%x
        call save_soln( &
            dat0((n - 1) * 4 + 1), &
            dat1((n - 1) * 4 + 1) &
        )
    end do
end subroutine

subroutine op2_k_airfoil_1_save_soln_main( &
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
    real(8), dimension(:), pointer, device :: dat1_d

    real(8) :: start_time, end_time
    real(4) :: transfer

    integer(4) :: num_blocks, max_blocks, block_size, block_limit
    integer(4) :: start, end

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("airfoil_1_save_soln", "CUDA", "Direct")
    call op_timing2_enter("Init")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges_grouped(set%setcptr, size(args), args, 2)

    if (n_exec == 0) then
        call op_timing2_exit()
        call op_timing2_exit()

        call op_mpi_wait_all_grouped(size(args), args, 2)
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
    call c_f_pointer(arg1%data_d, dat1_d, (/4 * getsetsizefromoparg(arg1)/))

    call op_timing2_next("Computation")
    start = 0
    end = set%setptr%size

    call op_timing2_enter("Kernel")
    call op2_k_airfoil_1_save_soln_wrapper<<<num_blocks, block_size>>>( &
        dat0_d, &
        dat1_d, &
        start, &
        end, &
        set%setptr%size &
    )

    call op_timing2_next("Process GBLs")
    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, "error in gpu kernel: ", "airfoil_1_save_soln"
        print *, cudaGetErrorString(err)
    end if

    call processDeviceGbls(args, size(args), block_size * max_blocks, block_size * max_blocks)

    call op_timing2_exit()
    call op_timing2_exit()

    call op_timing2_enter("Finalise")
    call op_mpi_set_dirtybit_cuda(size(args), args)

    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, cudaGetErrorString(err)
    end if

    call op_timing2_exit()
    call op_timing2_exit()
end subroutine

end module

module op2_m_airfoil_1_save_soln_fallback

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_1_save_soln_fallback

contains

SUBROUTINE save_soln(q, qold)
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
  REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: qold
  INTEGER(KIND = 4) :: i
  DO i = 1, 4
    qold(i) = q(i)
  END DO
END SUBROUTINE

subroutine op2_k_airfoil_1_save_soln_wrapper( &
    dat0, &
    dat1, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0
    real(8), dimension(:, :) :: dat1

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(2) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        call save_soln( &
            dat0(:, n), &
            dat1(:, n) &
        )
    end do
end subroutine

subroutine op2_k_airfoil_1_save_soln_fallback( &
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
    real(8), pointer, dimension(:, :) :: dat1

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("airfoil_1_save_soln", "seq", "Direct")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/4, getsetsizefromoparg(arg1)/))

    call op2_k_airfoil_1_save_soln_wrapper( &
        dat0, &
        dat1, &
        n_exec, &
        set, &
        args &
    )

    call op_timing2_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_timing2_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_timing2_exit()
end subroutine

end module

module op2_m_airfoil_1_save_soln

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_airfoil_1_save_soln_fallback
    use op2_m_airfoil_1_save_soln_main

    implicit none

    private
    public :: op2_k_airfoil_1_save_soln

contains

subroutine op2_k_airfoil_1_save_soln( &
    name, &
    set, &
    arg0, &
    arg1 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1

    if (op_check_whitelist("airfoil_1_save_soln")) then
        call op2_k_airfoil_1_save_soln_main( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    else
        call op2_k_airfoil_1_save_soln_fallback( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    end if
end subroutine

end module