module op2_m_reduction_1_cell_count

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_reduction_1_cell_count

contains

SUBROUTINE cell_count(res, cell_count_result)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(4) :: res
  INTEGER(KIND = 4), DIMENSION(1) :: cell_count_result
  INTEGER(KIND = 4) :: d
  DO d = 1, 4
    res(d) = 0.0
  END DO
  cell_count_result = cell_count_result + 1
END SUBROUTINE

subroutine op2_k_reduction_1_cell_count_wrapper( &
    dat0, &
    gbl1, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0

    integer(4), dimension(:) :: gbl1

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(2) :: args

    ! locals
    integer(4) :: n

    do n = 1, n_exec
        call cell_count( &
            dat0(:, n), &
            gbl1(1) &
        )
    end do
end subroutine

subroutine op2_k_reduction_1_cell_count( &
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

    integer(4), pointer, dimension(:) :: gbl1

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("reduction_1_cell_count", "seq", "Direct")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))

    call c_f_pointer(arg1%data, gbl1, (/1/))

    call op2_k_reduction_1_cell_count_wrapper( &
        dat0, &
        gbl1, &
        n_exec, &
        set, &
        args &
    )

    call op_timing2_next("MPI Wait")
    if ((n_exec == 0) .or. (n_exec == set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_timing2_next("MPI Reduce")

    call op_mpi_reduce_int(arg1, arg1%data)

    call op_timing2_exit()

    call op_mpi_set_dirtybit(size(args), args)
    call op_timing2_exit()
end subroutine

end module