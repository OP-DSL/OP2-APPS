#define op2_s(idx, stride) 1 + ((idx) - 1) * op2_stride_##stride##_d

module op2_m_jac1_mpi_1_res_kernel_m

    use cudafor
    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support
    use cudaconfigurationparams

    use op2_consts

    implicit none

    private
    public :: op2_k_jac1_mpi_1_res_kernel_m

contains

attributes(device) &
SUBROUTINE res_kernel(A, u, du, beta)
  IMPLICIT NONE
  REAL(KIND = 8), INTENT(IN) :: A
  REAL(KIND = 8), INTENT(IN) :: u
  REAL(KIND = 8), INTENT(INOUT) :: du
  REAL(KIND = 8), INTENT(IN) :: beta
  INTEGER(KIND = 4) :: op2_ret
  op2_ret = atomicAdd(du, 0.0D0 + beta * A * u)
END SUBROUTINE res_kernel

attributes(global) &
subroutine op2_k_jac1_mpi_1_res_kernel_wr( &
    dat0, &
    dat1, &
    dat2, &
    map0, &
    gbl3, &
    start, &
    end, &
    set_size &
)
    implicit none

    ! parameters
    real(8), dimension(*) :: dat0
    real(8), dimension(*) :: dat1
    real(8), dimension(*) :: dat2

    integer(4), dimension(*) :: map0

    real(8), dimension(*) :: gbl3

    integer(4), value :: start, end, set_size

    ! locals
    integer(4) :: thread_id, d, n, ret

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do n = thread_id + start, end, blockDim%x * gridDim%x
        call res_kernel( &
            dat0((n - 1) * 1 + 1), &
            dat1(map0(1 * set_size + n) * 1 + 1), &
            dat2(map0(0 * set_size + n) * 1 + 1), &
            gbl3(1) &
        )
    end do
end subroutine

subroutine op2_k_jac1_mpi_1_res_kernel_m( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3 &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3

    ! locals
    type(op_arg), dimension(4) :: args

    integer(4) :: n_exec, col, block, round, dim, err, d

    real(8), dimension(:), pointer, device :: dat0_d
    real(8), dimension(:), pointer, device :: dat1_d
    real(8), dimension(:), pointer, device :: dat2_d

    integer(4), dimension(:, :), pointer, device :: map0_d

    real(8), dimension(:), pointer :: gbl3
    real(8), dimension(:), pointer, device :: gbl3_d

    real(8) :: start_time, end_time
    real(4) :: transfer

    integer(4) :: num_blocks, max_blocks, block_size, block_limit
    integer(4) :: start, end
    integer(4), dimension(4) :: sections

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3

    call op_profile_enter_kernel("jac1_mpi_1_res_kernel", "CUDA", "Indirect (atomics)")
    call op_profile_enter("Init")

    call op_profile_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges_grouped(set%setcptr, size(args), args, 2)

    if (n_exec == 0) then
        call op_profile_exit()
        call op_profile_exit()

        call op_mpi_wait_all_grouped(size(args), args, 2)
        call op_mpi_set_dirtybit_cuda(size(args), args)
        err = cudaDeviceSynchronize()

        if (err /= 0) then
            print *, cudaGetErrorString(err)
        end if

        call op_profile_exit()
        return
    end if

    call op_profile_next("Update consts")
    call op_profile_exit()

    call setGblIncAtomic(logical(.false., c_bool))
    block_size = getBlockSize(name // c_null_char, set%setptr%size)
    block_limit = getBlockLimit(args, size(args), block_size, name // c_null_char)

    max_blocks = (max(set%setptr%core_size, &
        set%setptr%size + set%setptr%exec_size - set%setptr%core_size) - 1 + (block_size - 1)) / block_size
    max_blocks = min(max_blocks, block_limit)

    call op_profile_enter("Prepare GBLs")
    call prepareDeviceGbls(args, size(args), block_size * max_blocks)
    call op_profile_exit()

    arg0 = args(1)
    arg1 = args(2)
    arg2 = args(3)
    arg3 = args(4)

    call c_f_pointer(arg0%data_d, dat0_d, (/1 * round32f(getsetsizefromoparg(arg0)) /))
    call c_f_pointer(arg1%data_d, dat1_d, (/1 * round32f(getsetsizefromoparg(arg1)) /))
    call c_f_pointer(arg2%data_d, dat2_d, (/1 * round32f(getsetsizefromoparg(arg2)) /))

    call c_f_pointer(arg1%map_data_d, map0_d, (/set%setptr%size, getmapdimfromoparg(arg1)/))

    call c_f_pointer(arg3%data, gbl3, (/1/))
    call c_f_pointer(arg3%data_d, gbl3_d, (/1/))

    call op_profile_next("Computation")
    sections = (/0, set%setptr%core_size, set%setptr%size + set%setptr%exec_size, 0/)

    call op_profile_enter("Kernel")
    do round = 1, 2
        if (round == 2) then
            call op_profile_next("MPI Wait")
            call op_mpi_wait_all_grouped(size(args), args, 2)
            call op_profile_next("Kernel")
        end if

        start = sections(round)
        end = sections(round + 1)

        if (end - start > 0) then
            num_blocks = (end - start + (block_size - 1)) / block_size
            num_blocks = min(num_blocks, block_limit)

            call op2_k_jac1_mpi_1_res_kernel_wr<<<num_blocks, block_size>>>( &
                dat0_d, &
                dat1_d, &
                dat2_d, &
                map0_d, &
                gbl3_d, &
                start, &
                end, &
                round32f(set%setptr%size + set%setptr%exec_size) &
            )
        end if
    end do

    call op_profile_exit()
    call op_profile_exit()

    call op_profile_enter("Finalise")
    call op_mpi_set_dirtybit_cuda(size(args), args)

    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, cudaGetErrorString(err)
    end if

    call op_profile_exit()
    call op_profile_exit()
end subroutine

end module

module op2_m_jac1_mpi_1_res_kernel_fb

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_jac1_mpi_1_res_kernel_fb

contains

SUBROUTINE res_kernel(A, u, du, beta)
  IMPLICIT NONE
  REAL(KIND = 8), INTENT(IN) :: A
  REAL(KIND = 8), INTENT(IN) :: u
  REAL(KIND = 8), INTENT(INOUT) :: du
  REAL(KIND = 8), INTENT(IN) :: beta
  du = du + beta * A * u
END SUBROUTINE res_kernel

subroutine op2_k_jac1_mpi_1_res_kernel_wr( &
    dat0, &
    dat1, &
    dat2, &
    map0, &
    gbl3, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0
    real(8), dimension(:, :) :: dat1
    real(8), dimension(:, :) :: dat2

    integer(4), dimension(:, :) :: map0

    real(8), dimension(:) :: gbl3

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(4) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        if (n == set%setptr%core_size + 1) then
            call op_profile_next("MPI Wait")
            call op_mpi_wait_all(size(args), args)
            call op_profile_next("Computation")
        end if

        call res_kernel( &
            dat0(1, n), &
            dat1(1, map0(2, n) + 1), &
            dat2(1, map0(1, n) + 1), &
            gbl3(1) &
        )
    end do
end subroutine

subroutine op2_k_jac1_mpi_1_res_kernel_fb( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3 &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3

    ! locals
    type(op_arg), dimension(4) :: args

    integer(4) :: n_exec

    real(8), pointer, dimension(:, :) :: dat0
    real(8), pointer, dimension(:, :) :: dat1
    real(8), pointer, dimension(:, :) :: dat2

    integer(4), pointer, dimension(:, :) :: map0

    real(8), pointer, dimension(:) :: gbl3

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3

    call op_profile_enter_kernel("jac1_mpi_1_res_kernel", "seq", "Indirect")

    call op_profile_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_profile_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/1, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/1, getsetsizefromoparg(arg1)/))
    call c_f_pointer(arg2%data, dat2, (/1, getsetsizefromoparg(arg2)/))

    call c_f_pointer(arg1%map_data, map0, (/getmapdimfromoparg(arg1), set%setptr%size/))

    call c_f_pointer(arg3%data, gbl3, (/1/))

    call op2_k_jac1_mpi_1_res_kernel_wr( &
        dat0, &
        dat1, &
        dat2, &
        map0, &
        gbl3, &
        n_exec, &
        set, &
        args &
    )

    call op_profile_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_profile_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_profile_exit()
end subroutine

end module

module op2_m_jac1_mpi_1_res_kernel

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_jac1_mpi_1_res_kernel_fb
    use op2_m_jac1_mpi_1_res_kernel_m

    implicit none

    private
    public :: op2_k_jac1_mpi_1_res_kernel

contains

subroutine op2_k_jac1_mpi_1_res_kernel( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3

    if (op_check_whitelist("jac1_mpi_1_res_kernel")) then
        call op2_k_jac1_mpi_1_res_kernel_m( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3 &
        )
    else
        call op_check_fallback_mode("jac1_mpi_1_res_kernel")
        call op2_k_jac1_mpi_1_res_kernel_fb( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3 &
        )
    end if

end subroutine

end module