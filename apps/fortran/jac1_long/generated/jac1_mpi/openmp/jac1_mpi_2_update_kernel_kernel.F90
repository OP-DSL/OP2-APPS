
#define SIMD_LEN 8
#define op2_s(comp, simd_len) ((comp-1)*simd_len + 1)

module op2_m_jac1_mpi_2_update_kernel_m

    use iso_c_binding
    use omp_lib

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_jac1_mpi_2_update_kernel_m

contains

SUBROUTINE update_kernel_simd(r, du, u, u_sum, u_max)
  IMPLICIT NONE
  REAL(KIND = 8), INTENT(IN) :: r
  REAL(KIND = 8), INTENT(INOUT) :: du
  REAL(KIND = 8), INTENT(INOUT) :: u
  REAL(KIND = 8), INTENT(INOUT) :: u_sum
  REAL(KIND = 8), INTENT(INOUT) :: u_max
  u = u + du + op2_const_alpha * r
  du = 0.0_8
  u_sum = u_sum + u ** 2
  u_max = MAX(u_max, u)
END SUBROUTINE update_kernel_simd

SUBROUTINE update_kernel(r, du, u, u_sum, u_max)
  IMPLICIT NONE
  REAL(KIND = 8), INTENT(IN) :: r
  REAL(KIND = 8), INTENT(INOUT) :: du
  REAL(KIND = 8), INTENT(INOUT) :: u
  REAL(KIND = 8), INTENT(INOUT) :: u_sum
  REAL(KIND = 8), INTENT(INOUT) :: u_max
  u = u + du + op2_const_alpha * r
  du = 0.0_8
  u_sum = u_sum + u ** 2
  u_max = MAX(u_max, u)
END SUBROUTINE update_kernel

subroutine update_kernel_wrapper2( &
    dat0, &
    dat1, &
    dat2, &
    gbl3, &
    gbl4, &
    start, &
    end &
)
    implicit none

    ! parameters
    real(8), dimension(1, *) :: dat0
    real(8), dimension(1, *) :: dat1
    real(8), dimension(1, *) :: dat2

    real(8), dimension(1) :: gbl3
    real(8), dimension(1) :: gbl4

    integer(4) :: start, end

    ! locals
    integer(4) :: n
    integer(4) :: block, lane, d

    real(8), dimension(SIMD_LEN, 1) :: arg3_local
    real(8), dimension(SIMD_LEN, 1) :: arg4_local

    block = start
    do while (block + SIMD_LEN <= end)
        arg3_local = 0

        do lane = 1, SIMD_LEN
            n = block + lane - 1

            arg4_local(lane, :) = gbl4
        end do

        !$omp simd
        do lane = 1, SIMD_LEN
            n = block + lane - 1

            call update_kernel_simd( &
                dat0(1, n + 1), &
                dat1(1, n + 1), &
                dat2(1, n + 1), &
                arg3_local(lane, 1), &
                arg4_local(lane, 1) &
            )
        end do

        ! Reduction back to globals
        do lane = 1, SIMD_LEN
            n = block + lane - 1

            gbl3 = gbl3 + arg3_local(lane, :)
            gbl4 = MAX(gbl4, arg4_local(lane, :))
        end do

        block = block + SIMD_LEN
    end do

    do n = block, end
        call update_kernel( &
            dat0(1, n + 1), &
            dat1(1, n + 1), &
            dat2(1, n + 1), &
            gbl3(1), &
            gbl4(1) &
        )
    end do
end subroutine

subroutine update_kernel_wrapper( &
    name, &
    dat0, &
    dat1, &
    dat2, &
    gbl3, &
    gbl4, &
    set, &
    args, &
    num_dats_indirect, &
    dats_indirect &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name

    real(8), dimension(1, *) :: dat0
    real(8), dimension(1, *) :: dat1
    real(8), dimension(1, *) :: dat2

    real(8), dimension(1) :: gbl3
    real(8), dimension(1) :: gbl4

    type(op_set) :: set
    type(op_arg), dimension(5) :: args

    integer(4) :: num_dats_indirect
    integer(4), dimension(5) :: dats_indirect

    ! locals
    integer(4) :: thread, start, end, n
    integer(4) :: num_threads


    real(8), dimension(:), allocatable :: gbl3_temp
    real(8), dimension(:), allocatable :: gbl4_temp

    num_threads = omp_get_max_threads()

    allocate(gbl3_temp(num_threads * 64))
    gbl3_temp = 0

    allocate(gbl4_temp(num_threads * 64))

    do thread = 1, num_threads
        start = (thread - 1) * 64 + 1
        gbl4_temp(start : start + 0) = gbl4
    end do

    !$omp parallel do private(thread, start, end, n)
    do thread = 1, num_threads
        start = (set%setptr%size * (thread - 1)) / num_threads
        end = (set%setptr%size * thread) / num_threads - 1

        call update_kernel_wrapper2( &
            dat0, &
            dat1, &
            dat2, &
            gbl3_temp(omp_get_thread_num() * 64 + 1), &
            gbl4_temp(omp_get_thread_num() * 64 + 1), &
            start, &
            end &
        )
    end do

    do thread = 1, num_threads
        start = (thread - 1) * 64 + 1
        gbl3 = gbl3 + gbl3_temp(start : start + 0)
    end do

    do thread = 1, num_threads
        start = (thread - 1) * 64 + 1
        gbl4 = MAX(gbl4, gbl4_temp(start : start + 0))
    end do
end subroutine

subroutine op2_k_jac1_mpi_2_update_kernel_m( &
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

    integer(4) :: num_dats_indirect
    integer(4), dimension(5) :: dats_indirect

    integer(4) :: set_size

    real(8), pointer, dimension(:, :) :: dat0
    real(8), pointer, dimension(:, :) :: dat1
    real(8), pointer, dimension(:, :) :: dat2

    real(8), pointer, dimension(:) :: gbl3
    real(8), pointer, dimension(:) :: gbl4


    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4

    num_dats_indirect = 0
    dats_indirect = (/-1, -1, -1, -1, -1/)

    call op_profile_enter_kernel("jac1_mpi_2_update_kernel", "openmp", "Direct")

    call op_profile_enter("MPI Exchanges")
    set_size = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_profile_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/1, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/1, getsetsizefromoparg(arg1)/))
    call c_f_pointer(arg2%data, dat2, (/1, getsetsizefromoparg(arg2)/))

    call c_f_pointer(arg3%data, gbl3, (/1/))
    call c_f_pointer(arg4%data, gbl4, (/1/))

    call update_kernel_wrapper( &
        name, &
        dat0, &
        dat1, &
        dat2, &
        gbl3, &
        gbl4, &
        set, &
        args, &
        num_dats_indirect, &
        dats_indirect &
    )

    call op_profile_next("MPI Wait")
    if ((set_size .eq. 0) .or. (set_size .eq. set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_mpi_reduce_double(arg3, arg3%data)
    call op_mpi_reduce_double(arg4, arg4%data)

    call op_profile_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_profile_exit()
end subroutine

end module

module op2_m_jac1_mpi_2_update_kernel_fb

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_jac1_mpi_2_update_kernel_fb

contains

SUBROUTINE update_kernel(r, du, u, u_sum, u_max)
  IMPLICIT NONE
  REAL(KIND = 8), INTENT(IN) :: r
  REAL(KIND = 8), INTENT(INOUT) :: du
  REAL(KIND = 8), INTENT(INOUT) :: u
  REAL(KIND = 8), INTENT(INOUT) :: u_sum
  REAL(KIND = 8), INTENT(INOUT) :: u_max
  u = u + du + op2_const_alpha * r
  du = 0.0_8
  u_sum = u_sum + u ** 2
  u_max = MAX(u_max, u)
END SUBROUTINE update_kernel

subroutine op2_k_jac1_mpi_2_update_kernel_wr( &
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
        call update_kernel( &
            dat0(1, n), &
            dat1(1, n), &
            dat2(1, n), &
            gbl3(1), &
            gbl4(1) &
        )
    end do
end subroutine

subroutine op2_k_jac1_mpi_2_update_kernel_fb( &
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

    call op_profile_enter_kernel("jac1_mpi_2_update_kernel", "seq", "Direct")

    call op_profile_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_profile_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/1, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/1, getsetsizefromoparg(arg1)/))
    call c_f_pointer(arg2%data, dat2, (/1, getsetsizefromoparg(arg2)/))

    call c_f_pointer(arg3%data, gbl3, (/1/))
    call c_f_pointer(arg4%data, gbl4, (/1/))

    call op2_k_jac1_mpi_2_update_kernel_wr( &
        dat0, &
        dat1, &
        dat2, &
        gbl3, &
        gbl4, &
        n_exec, &
        set, &
        args &
    )

    call op_profile_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_profile_next("MPI Reduce")

    call op_mpi_reduce_double(arg3, arg3%data)
    call op_mpi_reduce_double(arg4, arg4%data)

    call op_profile_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_profile_exit()
end subroutine

end module

module op2_m_jac1_mpi_2_update_kernel

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_jac1_mpi_2_update_kernel_fb
    use op2_m_jac1_mpi_2_update_kernel_m

    implicit none

    private
    public :: op2_k_jac1_mpi_2_update_kernel

contains

subroutine op2_k_jac1_mpi_2_update_kernel( &
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

    if (op_check_whitelist("jac1_mpi_2_update_kernel")) then
        call op2_k_jac1_mpi_2_update_kernel_m( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4 &
        )
    else
        call op_check_fallback_mode("jac1_mpi_2_update_kernel")
        call op2_k_jac1_mpi_2_update_kernel_fb( &
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