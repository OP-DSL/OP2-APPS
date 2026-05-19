module op2_m_jac1_mpi_1_res_kernel_m

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    implicit none

    private
    public :: op2_k_jac1_mpi_1_res_kernel_m

    interface

        subroutine op2_k_jac1_mpi_1_res_kernel_m_c( &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3 &
        ) bind(C, name='op2_k_jac1_mpi_1_res_kernel_m_c')

            use iso_c_binding
            use op2_fortran_declarations

            type(c_ptr), value :: set

            type(op_arg), value :: arg0
            type(op_arg), value :: arg1
            type(op_arg), value :: arg2
            type(op_arg), value :: arg3

        end subroutine

    end interface

contains

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

    call op2_k_jac1_mpi_1_res_kernel_m_c( &
        set%setcptr, &
        arg0, &
        arg1, &
        arg2, &
        arg3 &
    )

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
            call op_timing2_next("MPI Wait")
            call op_mpi_wait_all(size(args), args)
            call op_timing2_next("Computation")
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

    call op_timing2_enter_kernel("jac1_mpi_1_res_kernel", "seq", "Indirect")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

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

    call op_timing2_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_timing2_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_timing2_exit()
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