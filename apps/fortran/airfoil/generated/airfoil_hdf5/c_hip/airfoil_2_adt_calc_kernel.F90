module op2_m_airfoil_2_adt_calc_main

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    implicit none

    private
    public :: op2_k_airfoil_2_adt_calc_main

    interface

        subroutine op2_k_airfoil_2_adt_calc_main_c( &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4, &
            arg5 &
        ) bind(C, name='op2_k_airfoil_2_adt_calc_main_c')

            use iso_c_binding
            use op2_fortran_declarations

            type(c_ptr), value :: set

            type(op_arg), value :: arg0
            type(op_arg), value :: arg1
            type(op_arg), value :: arg2
            type(op_arg), value :: arg3
            type(op_arg), value :: arg4
            type(op_arg), value :: arg5

        end subroutine

    end interface

contains

subroutine op2_k_airfoil_2_adt_calc_main( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4, &
    arg5 &
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
    type(op_arg) :: arg5

    call op2_k_airfoil_2_adt_calc_main_c( &
        set%setcptr, &
        arg0, &
        arg1, &
        arg2, &
        arg3, &
        arg4, &
        arg5 &
    )

end subroutine

end module

module op2_m_airfoil_2_adt_calc_fallback

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_2_adt_calc_fallback

contains

SUBROUTINE adt_calc(x1, x2, x3, x4, q, adt)
  REAL(KIND = 8), DIMENSION(2), INTENT(IN) :: x1, x2, x3, x4
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
  REAL(KIND = 8), INTENT(OUT) :: adt
  REAL(KIND = 8) :: dx, dy, ri, u, v, c
  ri = 1.0_8 / q(1)
  u = ri * q(2)
  v = ri * q(3)
  c = SQRT(op2_const_gam * op2_const_gm1 * (ri * q(4) - 0.5_8 * (u ** 2 + v ** 2)))
  dx = x2(1) - x1(1)
  dy = x2(2) - x1(2)
  adt = ABS(u * dy - v * dx) + c * SQRT(dx ** 2 + dy ** 2)
  dx = x3(1) - x2(1)
  dy = x3(2) - x2(2)
  adt = adt + ABS(u * dy - v * dx) + c * SQRT(dx ** 2 + dy ** 2)
  dx = x4(1) - x3(1)
  dy = x4(2) - x3(2)
  adt = adt + ABS(u * dy - v * dx) + c * SQRT(dx ** 2 + dy ** 2)
  dx = x1(1) - x4(1)
  dy = x1(2) - x4(2)
  adt = adt + ABS(u * dy - v * dx) + c * SQRT(dx ** 2 + dy ** 2)
  adt = adt / op2_const_cfl
END SUBROUTINE

subroutine op2_k_airfoil_2_adt_calc_wrapper( &
    dat0, &
    dat1, &
    dat2, &
    map0, &
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

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(6) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        if (n == set%setptr%core_size + 1) then
            call op_timing2_next("MPI Wait")
            call op_mpi_wait_all(size(args), args)
            call op_timing2_next("Computation")
        end if

        call adt_calc( &
            dat0(:, map0(1, n) + 1), &
            dat0(:, map0(2, n) + 1), &
            dat0(:, map0(3, n) + 1), &
            dat0(:, map0(4, n) + 1), &
            dat1(:, n), &
            dat2(1, n) &
        )
    end do
end subroutine

subroutine op2_k_airfoil_2_adt_calc_fallback( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4, &
    arg5 &
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
    type(op_arg) :: arg5

    ! locals
    type(op_arg), dimension(6) :: args

    integer(4) :: n_exec

    real(8), pointer, dimension(:, :) :: dat0
    real(8), pointer, dimension(:, :) :: dat1
    real(8), pointer, dimension(:, :) :: dat2

    integer(4), pointer, dimension(:, :) :: map0

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4
    args(6) = arg5

    call op_timing2_enter_kernel("airfoil_2_adt_calc", "seq", "Indirect")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/2, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg4%data, dat1, (/4, getsetsizefromoparg(arg4)/))
    call c_f_pointer(arg5%data, dat2, (/1, getsetsizefromoparg(arg5)/))

    call c_f_pointer(arg0%map_data, map0, (/getmapdimfromoparg(arg0), set%setptr%size/))

    call op2_k_airfoil_2_adt_calc_wrapper( &
        dat0, &
        dat1, &
        dat2, &
        map0, &
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

module op2_m_airfoil_2_adt_calc

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_airfoil_2_adt_calc_fallback
    use op2_m_airfoil_2_adt_calc_main

    implicit none

    private
    public :: op2_k_airfoil_2_adt_calc

contains

subroutine op2_k_airfoil_2_adt_calc( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4, &
    arg5 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3
    type(op_arg) :: arg4
    type(op_arg) :: arg5

    if (op_check_whitelist("airfoil_2_adt_calc")) then
        call op2_k_airfoil_2_adt_calc_main( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4, &
            arg5 &
        )
    else
        call op2_k_airfoil_2_adt_calc_fallback( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4, &
            arg5 &
        )
    end if
end subroutine

end module