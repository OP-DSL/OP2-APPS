module op2_m_airfoil_3_res_calc_main

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    implicit none

    private
    public :: op2_k_airfoil_3_res_calc_main

    interface

        subroutine op2_k_airfoil_3_res_calc_main_c( &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4, &
            arg5, &
            arg6, &
            arg7 &
        ) bind(C, name='op2_k_airfoil_3_res_calc_main_c')

            use iso_c_binding
            use op2_fortran_declarations

            type(c_ptr), value :: set

            type(op_arg), value :: arg0
            type(op_arg), value :: arg1
            type(op_arg), value :: arg2
            type(op_arg), value :: arg3
            type(op_arg), value :: arg4
            type(op_arg), value :: arg5
            type(op_arg), value :: arg6
            type(op_arg), value :: arg7

        end subroutine

    end interface

contains

subroutine op2_k_airfoil_3_res_calc_main( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4, &
    arg5, &
    arg6, &
    arg7 &
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
    type(op_arg) :: arg6
    type(op_arg) :: arg7

    call op2_k_airfoil_3_res_calc_main_c( &
        set%setcptr, &
        arg0, &
        arg1, &
        arg2, &
        arg3, &
        arg4, &
        arg5, &
        arg6, &
        arg7 &
    )

end subroutine

end module

module op2_m_airfoil_3_res_calc_fallback

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_3_res_calc_fallback

contains

SUBROUTINE res_calc(x1, x2, q1, q2, adt1, adt2, res1, res2)
  REAL(KIND = 8), DIMENSION(2), INTENT(IN) :: x1, x2
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q1, q2
  REAL(KIND = 8), INTENT(IN) :: adt1, adt2
  REAL(KIND = 8), DIMENSION(4), INTENT(INOUT) :: res1, res2
  REAL(KIND = 8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f
  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)
  ri = 1.0_8 / q1(1)
  p1 = op2_const_gm1 * (q1(4) - 0.5_8 * ri * (q1(2) ** 2 + q1(3) ** 2))
  vol1 = ri * (q1(2) * dy - q1(3) * dx)
  ri = 1.0_8 / q2(1)
  p2 = op2_const_gm1 * (q2(4) - 0.5_8 * ri * (q2(2) ** 2 + q2(3) ** 2))
  vol2 = ri * (q2(2) * dy - q2(3) * dx)
  mu = 0.5_8 * (adt1 + adt2) * op2_const_eps
  f = 0.5_8 * (vol1 * q1(1) + vol2 * q2(1)) + mu * (q1(1) - q2(1))
  res1(1) = res1(1) + f
  res2(1) = res2(1) - f
  f = 0.5_8 * (vol1 * q1(2) + p1 * dy + vol2 * q2(2) + p2 * dy) + mu * (q1(2) - q2(2))
  res1(2) = res1(2) + f
  res2(2) = res2(2) - f
  f = 0.5_8 * (vol1 * q1(3) - p1 * dx + vol2 * q2(3) - p2 * dx) + mu * (q1(3) - q2(3))
  res1(3) = res1(3) + f
  res2(3) = res2(3) - f
  f = 0.5_8 * (vol1 * (q1(4) + p1) + vol2 * (q2(4) + p2)) + mu * (q1(4) - q2(4))
  res1(4) = res1(4) + f
  res2(4) = res2(4) - f
END SUBROUTINE

subroutine op2_k_airfoil_3_res_calc_wrapper( &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    map0, &
    map1, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0
    real(8), dimension(:, :) :: dat1
    real(8), dimension(:, :) :: dat2
    real(8), dimension(:, :) :: dat3

    integer(4), dimension(:, :) :: map0
    integer(4), dimension(:, :) :: map1

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(8) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        if (n == set%setptr%core_size + 1) then
            call op_timing2_next("MPI Wait")
            call op_mpi_wait_all(size(args), args)
            call op_timing2_next("Computation")
        end if

        call res_calc( &
            dat0(:, map0(1, n) + 1), &
            dat0(:, map0(2, n) + 1), &
            dat1(:, map1(1, n) + 1), &
            dat1(:, map1(2, n) + 1), &
            dat2(1, map1(1, n) + 1), &
            dat2(1, map1(2, n) + 1), &
            dat3(:, map1(1, n) + 1), &
            dat3(:, map1(2, n) + 1) &
        )
    end do
end subroutine

subroutine op2_k_airfoil_3_res_calc_fallback( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4, &
    arg5, &
    arg6, &
    arg7 &
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
    type(op_arg) :: arg6
    type(op_arg) :: arg7

    ! locals
    type(op_arg), dimension(8) :: args

    integer(4) :: n_exec

    real(8), pointer, dimension(:, :) :: dat0
    real(8), pointer, dimension(:, :) :: dat1
    real(8), pointer, dimension(:, :) :: dat2
    real(8), pointer, dimension(:, :) :: dat3

    integer(4), pointer, dimension(:, :) :: map0
    integer(4), pointer, dimension(:, :) :: map1

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4
    args(6) = arg5
    args(7) = arg6
    args(8) = arg7

    call op_timing2_enter_kernel("airfoil_3_res_calc", "seq", "Indirect")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/2, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg2%data, dat1, (/4, getsetsizefromoparg(arg2)/))
    call c_f_pointer(arg4%data, dat2, (/1, getsetsizefromoparg(arg4)/))
    call c_f_pointer(arg6%data, dat3, (/4, getsetsizefromoparg(arg6)/))

    call c_f_pointer(arg0%map_data, map0, (/getmapdimfromoparg(arg0), set%setptr%size/))
    call c_f_pointer(arg2%map_data, map1, (/getmapdimfromoparg(arg2), set%setptr%size/))

    call op2_k_airfoil_3_res_calc_wrapper( &
        dat0, &
        dat1, &
        dat2, &
        dat3, &
        map0, &
        map1, &
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

module op2_m_airfoil_3_res_calc

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_airfoil_3_res_calc_fallback
    use op2_m_airfoil_3_res_calc_main

    implicit none

    private
    public :: op2_k_airfoil_3_res_calc

contains

subroutine op2_k_airfoil_3_res_calc( &
    name, &
    set, &
    arg0, &
    arg1, &
    arg2, &
    arg3, &
    arg4, &
    arg5, &
    arg6, &
    arg7 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1
    type(op_arg) :: arg2
    type(op_arg) :: arg3
    type(op_arg) :: arg4
    type(op_arg) :: arg5
    type(op_arg) :: arg6
    type(op_arg) :: arg7

    if (op_check_whitelist("airfoil_3_res_calc")) then
        call op2_k_airfoil_3_res_calc_main( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4, &
            arg5, &
            arg6, &
            arg7 &
        )
    else
        call op2_k_airfoil_3_res_calc_fallback( &
            name, &
            set, &
            arg0, &
            arg1, &
            arg2, &
            arg3, &
            arg4, &
            arg5, &
            arg6, &
            arg7 &
        )
    end if
end subroutine

end module