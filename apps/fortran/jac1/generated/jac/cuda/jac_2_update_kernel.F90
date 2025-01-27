#define op2_s(idx, stride) 1 + ((idx) - 1) * op2_stride_##stride##_d

module op2_m_jac_2_update_main

    use cudafor
    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support
    use cudaconfigurationparams

    use op2_consts

    implicit none

    private
    public :: op2_k_jac_2_update_main

    integer(4) :: op2_stride_gbl = 0
    integer(4), constant :: op2_stride_gbl_d = 0

contains

attributes(device) &
SUBROUTINE update(r, du, u, u_sum, u_max)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(1) :: r, du, u
  REAL(KIND = 8), DIMENSION(*) :: u_sum
  REAL(KIND = 8), DIMENSION(*) :: u_max
  u(1) = u(1) + du(1) + op2_const_alpha_d * r(1)
  du(1) = 0.0
  u_sum(op2_s(1, gbl)) = u_sum(op2_s(1, gbl)) + u(1) ** 2
  u_max(op2_s(1, gbl)) = MAX(u_max(op2_s(1, gbl)), u(1))
END SUBROUTINE

attributes(global) &
subroutine op2_k_jac_2_update_wrapper( &
    dat0, &
    dat1, &
    dat2, &
    gbl3, &
    gbl4, &
    start, &
    end, &
    set_size &
)
    implicit none

    ! parameters
    real(8), dimension(*) :: dat0
    real(8), dimension(*) :: dat1
    real(8), dimension(*) :: dat2

    real(8), dimension(*) :: gbl3
    real(8), dimension(*) :: gbl4

    integer(4), value :: start, end, set_size

    ! locals
    integer(4) :: thread_id, d, n, ret

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do n = thread_id + start, end, blockDim%x * gridDim%x
        call update( &
            dat0((n - 1) * 1 + 1), &
            dat1((n - 1) * 1 + 1), &
            dat2((n - 1) * 1 + 1), &
            gbl3(thread_id), &
            gbl4(thread_id) &
        )
    end do
end subroutine

attributes(global) &
subroutine op2_k_jac_2_update_init_gbls( &
    gbl3, &
    gbl4, &
    gbl4_ref, &
    dummy &
)
    implicit none

    ! parameters
    real(8), dimension(*) :: gbl3
    real(8), dimension(*) :: gbl4

    real(8), dimension(*) :: gbl4_ref

    integer(4), value :: dummy

    ! locals
    integer(4) :: thread_id, d

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do d = 1, 1
        gbl3(thread_id + (d - 1) * op2_stride_gbl_d) = 0
    end do

    do d = 1, 1
        gbl4(thread_id + (d - 1) * op2_stride_gbl_d) = gbl4_ref(d)
    end do
end subroutine

subroutine op2_k_jac_2_update_main( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4 &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3
    type(op_arg) :: arg4

    ! locals
    type(op_arg), dimension(5) :: args

    integer(4) :: n_exec, col, block, round, dim, err, d

    real(8), dimension(:), pointer, device :: dat0_d
    real(8), dimension(:), pointer, device :: dat1_d
    real(8), dimension(:), pointer, device :: dat2_d

    real(8), dimension(:), pointer :: gbl3
    real(8), dimension(:), pointer, device :: gbl3_d

    real(8), dimension(:), pointer :: gbl4
    real(8), dimension(:), pointer, device :: gbl4_d
    real(8), dimension(:), allocatable, device, save :: gbl4_ref_d

    real(8) :: start_time, end_time
    real(4) :: transfer

    integer(4) :: num_blocks, max_blocks, block_size, block_limit
    integer(4) :: start, end

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4

    call op_timing2_enter_kernel("jac_2_update", "CUDA", "Direct")
    call op_timing2_enter("Init")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges_grouped(set%setcptr, size(args), args, 2)

    if (n_exec == 0) then
        call op_timing2_exit()
        call op_timing2_exit()

        call op_mpi_wait_all_grouped(size(args), args, 2)
        call op_mpi_reduce_double(arg3, arg3%data)
        call op_mpi_reduce_double(arg4, arg4%data)
        call op_mpi_set_dirtybit_cuda(size(args), args)
        err = cudaDeviceSynchronize()

        if (err /= 0) then
            print *, cudaGetErrorString(err)
        end if

        call op_timing2_exit()
        return
    end if

    call op_timing2_next("Update consts")
    call op_update_const_cuda_alpha()

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
    arg2 = args(3)
    arg3 = args(4)
    arg4 = args(5)

    call c_f_pointer(arg0%data_d, dat0_d, (/1 * getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data_d, dat1_d, (/1 * getsetsizefromoparg(arg1)/))
    call c_f_pointer(arg2%data_d, dat2_d, (/1 * getsetsizefromoparg(arg2)/))

    call c_f_pointer(arg3%data, gbl3, (/1/))
    call c_f_pointer(arg3%data_d, gbl3_d, (/1 * block_size * max_blocks/))
    call c_f_pointer(arg4%data, gbl4, (/1/))
    call c_f_pointer(arg4%data_d, gbl4_d, (/1 * block_size * max_blocks/))

    if (op2_stride_gbl /= block_size * max_blocks) then
        op2_stride_gbl = block_size * max_blocks
        op2_stride_gbl_d = op2_stride_gbl
    end if

    if (.not. allocated(gbl4_ref_d)) then
        allocate(gbl4_ref_d(1))
    end if

    gbl4_ref_d = gbl4

    call op_timing2_enter("Init GBLs")
    call op2_k_jac_2_update_init_gbls<<<max_blocks, block_size>>>( &
        gbl3_d, &
        gbl4_d, &
        gbl4_ref_d, &
        0 &
    )

    call op_timing2_exit()
    call op_timing2_next("Computation")
    start = 0
    end = set%setptr%size

    call op_timing2_enter("Kernel")
    call op2_k_jac_2_update_wrapper<<<num_blocks, block_size>>>( &
        dat0_d, &
        dat1_d, &
        dat2_d, &
        gbl3_d, &
        gbl4_d, &
        start, &
        end, &
        set%setptr%size &
    )


    call op_timing2_next("Process GBLs")
    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, "error in gpu kernel: ", "jac_2_update"
        print *, cudaGetErrorString(err)
    end if

    call processDeviceGbls(args, size(args), block_size * max_blocks, block_size * max_blocks)

    call op_timing2_exit()
    call op_timing2_exit()

    call op_timing2_enter("Finalise")
    call op_mpi_reduce_double(arg3, arg3%data)
    call op_mpi_reduce_double(arg4, arg4%data)
    call op_mpi_set_dirtybit_cuda(size(args), args)

    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, cudaGetErrorString(err)
    end if

    call op_timing2_exit()
    call op_timing2_exit()
end subroutine

end module

module op2_m_jac_2_update_fallback

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_jac_2_update_fallback

contains

SUBROUTINE update(r, du, u, u_sum, u_max)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(1) :: r, du, u, u_sum, u_max
  u(1) = u(1) + du(1) + op2_const_alpha * r(1)
  du(1) = 0.0
  u_sum(1) = u_sum(1) + u(1) ** 2
  u_max(1) = MAX(u_max(1), u(1))
END SUBROUTINE

subroutine op2_k_jac_2_update_wrapper( &
    dat0, &
    dat1, &
    dat2, &
    gbl3, &
    gbl4, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0
    real(8), dimension(:, :) :: dat1
    real(8), dimension(:, :) :: dat2

    real(8), dimension(:) :: gbl3
    real(8), dimension(:) :: gbl4

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(5) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        call update( &
            dat0(1, n), &
            dat1(1, n), &
            dat2(1, n), &
            gbl3(1), &
            gbl4(1) &
        )
    end do
end subroutine

subroutine op2_k_jac_2_update_fallback( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4 &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3
    type(op_arg) :: arg4

    ! locals
    type(op_arg), dimension(5) :: args

    integer(4) :: n_exec

    real(8), pointer, dimension(:, :) :: dat0
    real(8), pointer, dimension(:, :) :: dat1
    real(8), pointer, dimension(:, :) :: dat2

    real(8), pointer, dimension(:) :: gbl3
    real(8), pointer, dimension(:) :: gbl4

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4

    call op_timing2_enter_kernel("jac_2_update", "seq", "Direct")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/1, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/1, getsetsizefromoparg(arg1)/))
    call c_f_pointer(arg2%data, dat2, (/1, getsetsizefromoparg(arg2)/))

    call c_f_pointer(arg3%data, gbl3, (/1/))
    call c_f_pointer(arg4%data, gbl4, (/1/))

    call op2_k_jac_2_update_wrapper( &
        dat0, &
        dat1, &
        dat2, &
        gbl3, &
        gbl4, &
        n_exec, &
        set, &
        args &
    )

    call op_timing2_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_timing2_next("MPI Reduce")

    call op_mpi_reduce_double(arg3, arg3%data)
    call op_mpi_reduce_double(arg4, arg4%data)

    call op_timing2_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_timing2_exit()
end subroutine

end module

module op2_m_jac_2_update

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_jac_2_update_fallback
    use op2_m_jac_2_update_main

    implicit none

    private
    public :: op2_k_jac_2_update

contains

subroutine op2_k_jac_2_update( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3
    type(op_arg) :: arg4

    if (op_check_whitelist("jac_2_update")) then
        call op2_k_jac_2_update_main( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4 &
        )
    else
        call op2_k_jac_2_update_fallback( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4 &
        )
    end if
end subroutine

end module