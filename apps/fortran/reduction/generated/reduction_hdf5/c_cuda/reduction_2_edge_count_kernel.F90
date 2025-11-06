module op2_m_reduction_2_edge_count_m

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    implicit none

    private
    public :: op2_k_reduction_2_edge_count_m

    interface

        subroutine op2_k_reduction_2_edge_count_m_c( &
            set, &
            arg0, &
            arg1 &
        ) bind(C, name='op2_k_reduction_2_edge_count_m_c')

            use iso_c_binding
            use op2_fortran_declarations

            type(c_ptr), value :: set

            type(op_arg), value :: arg0
            type(op_arg), value :: arg1

        end subroutine

    end interface

contains

subroutine op2_k_reduction_2_edge_count_m( &
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

    call op2_k_reduction_2_edge_count_m_c( &
        set%setcptr, &
        arg0, &
        arg1 &
    )

end subroutine

end module

module op2_m_reduction_2_edge_count_fb

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_reduction_2_edge_count_fb

contains

SUBROUTINE edge_count(res, edge_count_result)
  IMPLICIT NONE
  REAL(KIND = 8), DIMENSION(4) :: res
  INTEGER(KIND = 4), DIMENSION(1) :: edge_count_result
  INTEGER(KIND = 4) :: d
  DO d = 1, 4
    res(d) = 0.0
  END DO
  edge_count_result = edge_count_result + 1
END SUBROUTINE

subroutine op2_k_reduction_2_edge_count_wrapper( &
    dat0, &
    map0, &
    gbl1, &
    n_exec, &
    set, &
    args &
)
    implicit none

    ! parameters
    real(8), dimension(:, :) :: dat0

    integer(4), dimension(:, :) :: map0

    integer(4), dimension(:) :: gbl1

    integer(4) :: n_exec
    type(op_set) :: set
    type(op_arg), dimension(2) :: args

    ! locals
    integer(4), dimension(size(gbl1)) :: gbl1_temp

    integer(4) :: n

    gbl1_temp = gbl1

    do n = 1, n_exec
        if (n == set%setptr%core_size + 1) then
            call op_timing2_next("MPI Wait")
            call op_mpi_wait_all(size(args), args)
            call op_timing2_next("Computation")
        end if

        if (n == set%setptr%size + 1) then
            gbl1 = gbl1_temp
        end if

        call edge_count( &
            dat0(:, map0(1, n) + 1), &
            gbl1_temp(1) &
        )
    end do

    if (n_exec <= set%setptr%size) then
        gbl1 = gbl1_temp
    end if
end subroutine

subroutine op2_k_reduction_2_edge_count_fb( &
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

    integer(4), pointer, dimension(:, :) :: map0

    integer(4), pointer, dimension(:) :: gbl1

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1

    call op_timing2_enter_kernel("reduction_2_edge_count", "seq", "Indirect")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/4, getsetsizefromoparg(arg0)/))

    call c_f_pointer(arg0%map_data, map0, (/getmapdimfromoparg(arg0), set%setptr%size/))

    call c_f_pointer(arg1%data, gbl1, (/1/))

    call op2_k_reduction_2_edge_count_wrapper( &
        dat0, &
        map0, &
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

module op2_m_reduction_2_edge_count

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_reduction_2_edge_count_fb
    use op2_m_reduction_2_edge_count_m

    implicit none

    private
    public :: op2_k_reduction_2_edge_count

contains

subroutine op2_k_reduction_2_edge_count( &
    name, &
    set, &
    arg0, &
    arg1 &
)
    character(kind=c_char, len=*) :: name
    type(op_set) :: set

    type(op_arg) :: arg0
    type(op_arg) :: arg1

    if (op_check_whitelist("reduction_2_edge_count")) then
        call op2_k_reduction_2_edge_count_m( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    else
        call op_check_fallback_mode("reduction_2_edge_count")
        call op2_k_reduction_2_edge_count_fb( &
            name, &
            set, &
            arg0, &
            arg1 &
        )
    end if

end subroutine

end module