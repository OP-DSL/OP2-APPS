module op2_m_airfoil_5_update

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_5_update

contains

SUBROUTINE update(qold, q, res, adt, rms, maxerr, idx, errloc)
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: qold
  REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: q
  REAL(KIND = 8), DIMENSION(4), INTENT(INOUT) :: res
  REAL(KIND = 8), INTENT(IN) :: adt
  REAL(KIND = 8), DIMENSION(2), INTENT(INOUT) :: rms
  REAL(KIND = 8), INTENT(INOUT) :: maxerr
  INTEGER(KIND = 4), INTENT(IN) :: idx
  INTEGER(KIND = 4), INTENT(OUT) :: errloc
  REAL(KIND = 8) :: del, adti
  INTEGER(KIND = 4) :: i
  adti = 1.0_8 / adt
  DO i = 1, 4
    del = adti * res(i)
    q(i) = qold(i) - del
    res(i) = 0.0_8
    rms(2) = rms(2) + del ** 2
    IF (del ** 2 > maxerr) THEN
      maxerr = del ** 2
      errloc = idx
    END IF
  END DO
END SUBROUTINE

subroutine op2_k_airfoil_5_update_wrapper( &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    gbl4, &
    gbl5, &
    info7, &
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

    real(8), dimension(:) :: gbl4
    real(8), dimension(:) :: gbl5

    integer(4), dimension(:) :: info7

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(8) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        call update( &
            dat0(:, n), &
            dat1(:, n), &
            dat2(:, n), &
            dat3(1, n), &
            gbl4, &
            gbl5(1), &
            n, &
            info7(1) &
        )
    end do
end subroutine

subroutine op2_k_airfoil_5_update( &
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

    real(8), pointer, dimension(:) :: gbl4
    real(8), pointer, dimension(:) :: gbl5

    integer(4), pointer, dimension(:) :: info7

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4
    args(6) = arg5
    args(7) = arg6
    args(8) = arg7

    call op_timing2_enter_kernel("airfoil_5_update", "seq", "Direct")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/4, getsetsizefromoparg(arg1)/))
    call c_f_pointer(arg2%data, dat2, (/4, getsetsizefromoparg(arg2)/))
    call c_f_pointer(arg3%data, dat3, (/1, getsetsizefromoparg(arg3)/))

    call c_f_pointer(arg4%data, gbl4, (/2/))
    call c_f_pointer(arg5%data, gbl5, (/1/))

    call c_f_pointer(arg7%data, info7, (/1/))

    call op2_k_airfoil_5_update_wrapper( &
        dat0, &
        dat1, &
        dat2, &
        dat3, &
        gbl4, &
        gbl5, &
        info7, &
        n_exec, &
        set, &
        args &
    )

    call op_timing2_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_timing2_next("MPI Reduce")

    call op_mpi_reduce_double(arg4, arg4%data)
    call op_mpi_reduce_double(arg5, arg5%data)

    call op_timing2_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_timing2_exit()
end subroutine

end module