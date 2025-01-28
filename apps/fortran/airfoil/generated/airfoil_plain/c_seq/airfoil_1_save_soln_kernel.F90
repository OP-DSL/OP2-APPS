module op2_m_airfoil_1_save_soln_main

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    implicit none

    private
    public :: op2_k_airfoil_1_save_soln_main

    interface

        subroutine op2_k_airfoil_1_save_soln_main_c( &
            set, &
            arg0, &
            arg1 &
        ) bind(C, name='op2_k_airfoil_1_save_soln_main_c')

            use iso_c_binding
            use op2_fortran_declarations

            type(c_ptr), value :: set

            type(op_arg), value :: arg0
            type(op_arg), value :: arg1

        end subroutine

    end interface

contains

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

    call op2_k_airfoil_1_save_soln_main_c( &
        set%setcptr, &
        arg0, &
        arg1 &
    )

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