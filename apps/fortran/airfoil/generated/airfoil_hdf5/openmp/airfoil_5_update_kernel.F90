
#define SIMD_LEN 8
#define op2_s(comp, simd_len) ((comp-1)*simd_len + 1)

module op2_m_airfoil_5_update_m

    use iso_c_binding
    use omp_lib

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_5_update_m

contains

SUBROUTINE update_simd(qold, q, res, adt, rms, maxerr, idx, errloc)
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: qold
  REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: q
  REAL(KIND = 8), DIMENSION(4), INTENT(INOUT) :: res
  REAL(KIND = 8), INTENT(IN) :: adt
  REAL(KIND = 8), DIMENSION(*), INTENT(INOUT) :: rms
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
    rms(op2_s(2, SIMD_LEN)) = rms(op2_s(2, SIMD_LEN)) + del ** 2
    IF (del ** 2 > maxerr) THEN
      maxerr = del ** 2
      errloc = idx
    END IF
  END DO
END SUBROUTINE

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

subroutine update_wrapper2( &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    gbl4, &
    gbl5, &
    info7, &
    start, &
    end &
)
    implicit none

    ! parameters
    real(8), dimension(4, *) :: dat0
    real(8), dimension(4, *) :: dat1
    real(8), dimension(4, *) :: dat2
    real(8), dimension(1, *) :: dat3

    real(8), dimension(2) :: gbl4
    real(8), dimension(1) :: gbl5

    integer(4), dimension(1) :: info7

    integer(4) :: start, end

    ! locals
    integer(4) :: n
    integer(4) :: block, lane, d

    real(8), dimension(SIMD_LEN, 2) :: arg4_local
    real(8), dimension(SIMD_LEN, 1) :: arg5_local
    integer(4), dimension(SIMD_LEN, 1) :: info7_local

    block = start
    do while (block + SIMD_LEN <= end)
        arg4_local = 0

        do lane = 1, SIMD_LEN
            n = block + lane - 1

            arg5_local(lane, :) = gbl5
        end do

        !$omp simd
        do lane = 1, SIMD_LEN
            n = block + lane - 1

            call update_simd( &
                dat0(:, n + 1), &
                dat1(:, n + 1), &
                dat2(:, n + 1), &
                dat3(1, n + 1), &
                arg4_local(lane, 1), &
                arg5_local(lane, 1), &
                n + 1, &
                info7_local(lane, 1) &
            )
        end do

        ! Reduction back to globals
        do lane = 1, SIMD_LEN
            n = block + lane - 1

            gbl4 = gbl4 + arg4_local(lane, :)
            where (arg5_local(lane, :) > gbl5(:))
                gbl5(:) = arg5_local(lane, :)
                info7(:) = info7_local(lane, :)
            end where
        end do

        block = block + SIMD_LEN
    end do

    do n = block, end
        call update( &
            dat0(:, n + 1), &
            dat1(:, n + 1), &
            dat2(:, n + 1), &
            dat3(1, n + 1), &
            gbl4, &
            gbl5(1), &
            n + 1, &
            info7(1) &
        )
    end do
end subroutine

subroutine update_wrapper( &
    name, &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    gbl4, &
    gbl5, &
    info7, &
    set, &
    args, &
    num_dats_indirect, &
    dats_indirect &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name

    real(8), dimension(4, *) :: dat0
    real(8), dimension(4, *) :: dat1
    real(8), dimension(4, *) :: dat2
    real(8), dimension(1, *) :: dat3

    real(8), dimension(2) :: gbl4
    real(8), dimension(1) :: gbl5

    integer(4), dimension(1) :: info7

    type(op_set) :: set
    type(op_arg), dimension(8) :: args

    integer(4) :: num_dats_indirect
    integer(4), dimension(8) :: dats_indirect

    ! locals
    integer(4) :: thread, start, end, n
    integer(4) :: num_threads


    real(8), dimension(:), allocatable :: gbl4_temp
    real(8), dimension(:), allocatable :: gbl5_temp

    integer(4), dimension(:), allocatable :: info7_temp

    num_threads = omp_get_max_threads()

    allocate(gbl4_temp(num_threads * 64))
    gbl4_temp = 0

    allocate(gbl5_temp(num_threads * 64))

    do thread = 1, num_threads
        start = (thread - 1) * 64 + 1
        gbl5_temp(start : start + 0) = gbl5
    end do

    allocate(info7_temp(num_threads * 64))

    !$omp parallel do private(thread, start, end, n)
    do thread = 1, num_threads
        start = (set%setptr%size * (thread - 1)) / num_threads
        end = (set%setptr%size * thread) / num_threads - 1

        call update_wrapper2( &
            dat0, &
            dat1, &
            dat2, &
            dat3, &
            gbl4_temp(omp_get_thread_num() * 64 + 1), &
            gbl5_temp(omp_get_thread_num() * 64 + 1), &
            info7_temp(omp_get_thread_num() * 64 + 1), &
            start, &
            end &
        )
    end do

    do thread = 1, num_threads
        start = (thread - 1) * 64 + 1
        gbl4 = gbl4 + gbl4_temp(start : start + 1)
    end do

    do thread = 1, num_threads
        start = (thread - 1) * 64 + 1
        where (gbl5_temp(start : start + 0) > gbl5(:))
            gbl5(:) = gbl5_temp(start : start + 0)
            info7(:) = info7_temp(start : start + 0)
        end where
    end do
end subroutine

subroutine op2_k_airfoil_5_update_m( &
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

    integer(4) :: num_dats_indirect
    integer(4), dimension(8) :: dats_indirect

    integer(4) :: set_size

    real(8), pointer, dimension(:, :) :: dat0
    real(8), pointer, dimension(:, :) :: dat1
    real(8), pointer, dimension(:, :) :: dat2
    real(8), pointer, dimension(:, :) :: dat3

    real(8), pointer, dimension(:) :: gbl4
    real(8), pointer, dimension(:) :: gbl5

    integer(4), pointer, dimension(:) :: info7

    real(8) :: start_time, end_time
    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4
    args(6) = arg5
    args(7) = arg6
    args(8) = arg7

    num_dats_indirect = 0
    dats_indirect = (/-1, -1, -1, -1, -1, -1, -1, -1/)

    call op_timers_core(start_time)
    set_size = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/4, getsetsizefromoparg(arg1)/))
    call c_f_pointer(arg2%data, dat2, (/4, getsetsizefromoparg(arg2)/))
    call c_f_pointer(arg3%data, dat3, (/1, getsetsizefromoparg(arg3)/))

    call c_f_pointer(arg4%data, gbl4, (/2/))
    call c_f_pointer(arg5%data, gbl5, (/1/))

    call c_f_pointer(arg7%data, info7, (/1/))

    call update_wrapper( &
        name, &
        dat0, &
        dat1, &
        dat2, &
        dat3, &
        gbl4, &
        gbl5, &
        info7, &
        set, &
        args, &
        num_dats_indirect, &
        dats_indirect &
    )

    if ((set_size .eq. 0) .or. (set_size .eq. set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_mpi_reduce_double(arg4, arg4%data)
    call op_mpi_reduce_double(arg5, arg5%data)

    call op_mpi_set_dirtybit(size(args), args)
    call op_timers_core(end_time)

    ! todo: review kernel transfer calculation
    transfer = 0.0

    call setkerneltime(5, name // c_null_char, end_time - start_time, transfer, 0.0, 1)
end subroutine

end module

module op2_m_airfoil_5_update_fb

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_5_update_fb

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

subroutine op2_k_airfoil_5_update_wr( &
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

subroutine op2_k_airfoil_5_update_fb( &
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

    call op2_k_airfoil_5_update_wr( &
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

module op2_m_airfoil_5_update

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_airfoil_5_update_fb
    use op2_m_airfoil_5_update_m

    implicit none

    private
    public :: op2_k_airfoil_5_update

contains

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

    if (op_check_whitelist("airfoil_5_update")) then
        call op2_k_airfoil_5_update_m( &
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
        call op_check_fallback_mode("airfoil_5_update")
        call op2_k_airfoil_5_update_fb( &
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