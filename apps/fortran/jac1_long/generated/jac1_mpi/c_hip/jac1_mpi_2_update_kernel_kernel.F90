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

    call op_timing2_enter_kernel("jac1_mpi_2_update_kernel", "seq", "Direct")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

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

module op2_m_jac1_mpi_2_update_kernel

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_jac1_mpi_2_update_kernel_fb

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

end subroutine

end module