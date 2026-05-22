
#define SIMD_LEN 8
#define op2_s(comp, simd_len) ((comp-1)*simd_len + 1)

module op2_m_airfoil_1_save_soln_m

    use iso_c_binding
    use omp_lib

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_1_save_soln_m

contains

SUBROUTINE save_soln_simd(q, qold)
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
  REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: qold
  INTEGER(KIND = 4) :: i
  DO i = 1, 4
    qold(i) = q(i)
  END DO
END SUBROUTINE

SUBROUTINE save_soln(q, qold)
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
  REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: qold
  INTEGER(KIND = 4) :: i
  DO i = 1, 4
    qold(i) = q(i)
  END DO
END SUBROUTINE

subroutine save_soln_wrapper2( &
    dat0, &
    dat1, &
    start, &
    end &
)
    implicit none

    ! parameters
    real(8), dimension(4, *) :: dat0
    real(8), dimension(4, *) :: dat1

    integer(4) :: start, end

    ! locals
    integer(4) :: n
    integer(4) :: block, lane, d


    block = start
    do while (block + SIMD_LEN <= end)

        do lane = 1, SIMD_LEN
            n = block + lane - 1

        end do

        !$omp simd
        do lane = 1, SIMD_LEN
            n = block + lane - 1

            call save_soln_simd( &
                dat0(:, n + 1), &
                dat1(:, n + 1) &
            )
        end do

        ! Reduction back to globals
        do lane = 1, SIMD_LEN
            n = block + lane - 1

        end do

        block = block + SIMD_LEN
    end do

    do n = block, end
        call save_soln( &
            dat0(:, n + 1), &
            dat1(:, n + 1) &
        )
    end do
end subroutine

subroutine save_soln_wrapper( &
    name, &
    dat0, &
    dat1, &
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

    type(op_set) :: set
    type(op_arg), dimension(2) :: args

    integer(4) :: num_dats_indirect
    integer(4), dimension(2) :: dats_indirect

    ! locals
    integer(4) :: thread, start, end, n
    integer(4) :: num_threads


    num_threads = omp_get_max_threads()

    !$omp parallel do private(thread, start, end, n)
    do thread = 1, num_threads
        start = (set%setptr%size * (thread - 1)) / num_threads
        end = (set%setptr%size * thread) / num_threads - 1

        call save_soln_wrapper2( &
            dat0, &
            dat1, &
            start, &
            end &
        )
    end do
end subroutine

subroutine op2_k_airfoil_1_save_soln_m( &
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

    integer(4) :: num_dats_indirect
    integer(4), dimension(2) :: dats_indirect

    integer(4) :: set_size

    real(8), pointer, dimension(:, :) :: dat0
    real(8), pointer, dimension(:, :) :: dat1


    args(1) = arg0
    args(2) = arg1

    num_dats_indirect = 0
    dats_indirect = (/-1, -1/)

    call op_profile_enter_kernel("airfoil_1_save_soln", "openmp", "Direct")

    call op_profile_enter("MPI Exchanges")
    set_size = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_profile_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/4, getsetsizefromoparg(arg1)/))

    call save_soln_wrapper( &
        name, &
        dat0, &
        dat1, &
        set, &
        args, &
        num_dats_indirect, &
        dats_indirect &
    )

    call op_profile_next("MPI Wait")
    if ((set_size .eq. 0) .or. (set_size .eq. set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_profile_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_profile_exit()
end subroutine

end module

module op2_m_airfoil_1_save_soln_fb

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_1_save_soln_fb

contains

SUBROUTINE save_soln(q, qold)
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
  REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: qold
  INTEGER(KIND = 4) :: i
  DO i = 1, 4
    qold(i) = q(i)
  END DO
END SUBROUTINE

subroutine op2_k_airfoil_1_save_soln_wr( &
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

subroutine op2_k_airfoil_1_save_soln_fb( &
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

    call op_profile_enter_kernel("airfoil_1_save_soln", "seq", "Direct")

    call op_profile_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_profile_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg1%data, dat1, (/4, getsetsizefromoparg(arg1)/))

    call op2_k_airfoil_1_save_soln_wr( &
        dat0, &
        dat1, &
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

module op2_m_airfoil_1_save_soln

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_airfoil_1_save_soln_fb
    use op2_m_airfoil_1_save_soln_m

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
        call op2_k_airfoil_1_save_soln_m( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    else
        call op_check_fallback_mode("airfoil_1_save_soln")
        call op2_k_airfoil_1_save_soln_fb( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    end if

end subroutine

end module