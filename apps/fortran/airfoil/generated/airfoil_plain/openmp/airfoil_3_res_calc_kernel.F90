
#define SIMD_LEN 8
#define op2_s(comp, simd_len) ((comp-1)*simd_len + 1)

module op2_m_airfoil_3_res_calc_m

    use iso_c_binding
    use omp_lib

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_3_res_calc_m

contains

SUBROUTINE res_calc_simd(x1, x2, q1, q2, adt1, adt2, res1, res2)
  REAL(KIND = 8), DIMENSION(*), INTENT(IN) :: x2
  REAL(KIND = 8), DIMENSION(*), INTENT(IN) :: q2
  REAL(KIND = 8), INTENT(IN) :: adt1, adt2
  REAL(KIND = 8), DIMENSION(*), INTENT(INOUT) :: res2
  REAL(KIND = 8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f
  REAL(KIND = 8), DIMENSION(*), INTENT(IN) :: x1
  REAL(KIND = 8), DIMENSION(*), INTENT(IN) :: q1
  REAL(KIND = 8), DIMENSION(*), INTENT(INOUT) :: res1
  dx = x1(op2_s(1, SIMD_LEN)) - x2(op2_s(1, SIMD_LEN))
  dy = x1(op2_s(2, SIMD_LEN)) - x2(op2_s(2, SIMD_LEN))
  ri = 1.0_8 / q1(op2_s(1, SIMD_LEN))
  p1 = op2_const_gm1 * (q1(op2_s(4, SIMD_LEN)) - 0.5_8 * ri * (q1(op2_s(2, SIMD_LEN)) ** 2 + q1(op2_s(3, SIMD_LEN)) ** 2))
  vol1 = ri * (q1(op2_s(2, SIMD_LEN)) * dy - q1(op2_s(3, SIMD_LEN)) * dx)
  ri = 1.0_8 / q2(op2_s(1, SIMD_LEN))
  p2 = op2_const_gm1 * (q2(op2_s(4, SIMD_LEN)) - 0.5_8 * ri * (q2(op2_s(2, SIMD_LEN)) ** 2 + q2(op2_s(3, SIMD_LEN)) ** 2))
  vol2 = ri * (q2(op2_s(2, SIMD_LEN)) * dy - q2(op2_s(3, SIMD_LEN)) * dx)
  mu = 0.5_8 * (adt1 + adt2) * op2_const_eps
  f = 0.5_8 * (vol1 * q1(op2_s(1, SIMD_LEN)) + vol2 * q2(op2_s(1, SIMD_LEN))) + mu * (q1(op2_s(1, SIMD_LEN)) - q2(op2_s(1, SIMD_LEN)))
  res1(op2_s(1, SIMD_LEN)) = res1(op2_s(1, SIMD_LEN)) + f
  res2(op2_s(1, SIMD_LEN)) = res2(op2_s(1, SIMD_LEN)) - f
  f = 0.5_8 * (vol1 * q1(op2_s(2, SIMD_LEN)) + p1 * dy + vol2 * q2(op2_s(2, SIMD_LEN)) + p2 * dy) + mu * (q1(op2_s(2, SIMD_LEN)) - q2(op2_s(2, SIMD_LEN)))
  res1(op2_s(2, SIMD_LEN)) = res1(op2_s(2, SIMD_LEN)) + f
  res2(op2_s(2, SIMD_LEN)) = res2(op2_s(2, SIMD_LEN)) - f
  f = 0.5_8 * (vol1 * q1(op2_s(3, SIMD_LEN)) - p1 * dx + vol2 * q2(op2_s(3, SIMD_LEN)) - p2 * dx) + mu * (q1(op2_s(3, SIMD_LEN)) - q2(op2_s(3, SIMD_LEN)))
  res1(op2_s(3, SIMD_LEN)) = res1(op2_s(3, SIMD_LEN)) + f
  res2(op2_s(3, SIMD_LEN)) = res2(op2_s(3, SIMD_LEN)) - f
  f = 0.5_8 * (vol1 * (q1(op2_s(4, SIMD_LEN)) + p1) + vol2 * (q2(op2_s(4, SIMD_LEN)) + p2)) + mu * (q1(op2_s(4, SIMD_LEN)) - q2(op2_s(4, SIMD_LEN)))
  res1(op2_s(4, SIMD_LEN)) = res1(op2_s(4, SIMD_LEN)) + f
  res2(op2_s(4, SIMD_LEN)) = res2(op2_s(4, SIMD_LEN)) - f
END SUBROUTINE

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

subroutine res_calc_wrapper2( &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    map0, &
    map1, &
    start, &
    end &
)
    implicit none

    ! parameters
    real(8), dimension(2, *) :: dat0
    real(8), dimension(4, *) :: dat1
    real(8), dimension(1, *) :: dat2
    real(8), dimension(4, *) :: dat3

    integer(4), dimension(:, :) :: map0
    integer(4), dimension(:, :) :: map1

    integer(4) :: start, end

    ! locals
    integer(4) :: n
    integer(4) :: block, lane, d

    real(8), dimension(SIMD_LEN, 2) :: arg0_local
    real(8), dimension(SIMD_LEN, 2) :: arg1_local
    real(8), dimension(SIMD_LEN, 4) :: arg2_local
    real(8), dimension(SIMD_LEN, 4) :: arg3_local
    real(8), dimension(SIMD_LEN, 1) :: arg4_local
    real(8), dimension(SIMD_LEN, 1) :: arg5_local
    real(8), dimension(SIMD_LEN, 4) :: arg6_local
    real(8), dimension(SIMD_LEN, 4) :: arg7_local

    block = start
    do while (block + SIMD_LEN <= end)
        arg6_local = 0
        arg7_local = 0

        do lane = 1, SIMD_LEN
            n = block + lane - 1

            arg0_local(lane, :) = dat0(:, map0(1, n + 1) + 1)
            arg1_local(lane, :) = dat0(:, map0(2, n + 1) + 1)
            arg2_local(lane, :) = dat1(:, map1(1, n + 1) + 1)
            arg3_local(lane, :) = dat1(:, map1(2, n + 1) + 1)
            arg4_local(lane, :) = dat2(:, map1(1, n + 1) + 1)
            arg5_local(lane, :) = dat2(:, map1(2, n + 1) + 1)
        end do

        !$omp simd
        do lane = 1, SIMD_LEN
            n = block + lane - 1

            call res_calc_simd( &
                arg0_local(lane, 1), &
                arg1_local(lane, 1), &
                arg2_local(lane, 1), &
                arg3_local(lane, 1), &
                arg4_local(lane, 1), &
                arg5_local(lane, 1), &
                arg6_local(lane, 1), &
                arg7_local(lane, 1) &
            )
        end do

        ! Reduction back to globals
        do lane = 1, SIMD_LEN
            n = block + lane - 1

            dat3(:, map1(1, n + 1) + 1) = dat3(:, map1(1, n + 1) + 1) + arg6_local(lane, :)
            dat3(:, map1(2, n + 1) + 1) = dat3(:, map1(2, n + 1) + 1) + arg7_local(lane, :)
        end do

        block = block + SIMD_LEN
    end do

    do n = block, end
        call res_calc( &
            dat0(:, map0(1, n + 1) + 1), &
            dat0(:, map0(2, n + 1) + 1), &
            dat1(:, map1(1, n + 1) + 1), &
            dat1(:, map1(2, n + 1) + 1), &
            dat2(1, map1(1, n + 1) + 1), &
            dat2(1, map1(2, n + 1) + 1), &
            dat3(:, map1(1, n + 1) + 1), &
            dat3(:, map1(2, n + 1) + 1) &
        )
    end do
end subroutine

subroutine res_calc_wrapper( &
    name, &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    map0, &
    map1, &
    set, &
    args, &
    num_dats_indirect, &
    dats_indirect &
)
    implicit none

    ! parameters
    character(kind=c_char, len=*) :: name

    real(8), dimension(2, *) :: dat0
    real(8), dimension(4, *) :: dat1
    real(8), dimension(1, *) :: dat2
    real(8), dimension(4, *) :: dat3

    integer(4), dimension(:, :) :: map0
    integer(4), dimension(:, :) :: map1

    type(op_set) :: set
    type(op_arg), dimension(8) :: args

    integer(4) :: num_dats_indirect
    integer(4), dimension(8) :: dats_indirect

    ! locals
    integer(4) :: thread, start, end, n
    integer(4) :: num_threads

    integer(4) :: part_size, col, block_idx, block_offset, num_blocks, block_id, num_elem, offset

    type(op_plan), pointer :: plan
    integer(4), dimension(:), pointer :: plan_ncolblk, plan_blkmap, plan_nelems, plan_offset

#ifdef OP_PART_SIZE_3
    part_size = OP_PART_SIZE_3
#else
    part_size = 0
#endif

    plan => fortranplancaller( &
        name // c_null_char, &
        set%setcptr, &
        part_size, &
        size(args), &
        args, &
        num_dats_indirect, &
        dats_indirect, &
        2 &
    )

    call c_f_pointer(plan%ncolblk, plan_ncolblk, (/ plan%ncolors /))
    call c_f_pointer(plan%blkmap, plan_blkmap, (/ plan%nblocks /))
    call c_f_pointer(plan%nelems, plan_nelems, (/ plan%nblocks /))
    call c_f_pointer(plan%offset, plan_offset, (/ plan%nblocks /))

    block_offset = 0
    do col = 1, plan%ncolors
        if (col == plan%ncolors_core + 1) then
            call op_mpi_wait_all(size(args), args)
        end if

        num_blocks = plan_ncolblk(col)

        !$omp parallel do private(thread, block_idx, block_id, num_elem, offset, n)
        do block_idx = 1, num_blocks
            thread = omp_get_thread_num() + 1

            block_id = plan_blkmap(block_idx + block_offset) + 1
            num_elem = plan_nelems(block_id)
            offset = plan_offset(block_id)

            call res_calc_wrapper2( &
                dat0, &
                dat1, &
                dat2, &
                dat3, &
                map0, &
                map1, &
                offset, &
                offset + num_elem - 1 &
            )
        end do

        block_offset = block_offset + num_blocks
    end do
end subroutine

subroutine op2_k_airfoil_3_res_calc_m( &
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

    integer(4), pointer, dimension(:, :) :: map0
    integer(4), pointer, dimension(:, :) :: map1

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

    num_dats_indirect = 4
    dats_indirect = (/0, 0, 1, 1, 2, 2, 3, 3/)

    call op_timers_core(start_time)
    set_size = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call c_f_pointer(arg0%data, dat0, (/2, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg2%data, dat1, (/4, getsetsizefromoparg(arg2)/))
    call c_f_pointer(arg4%data, dat2, (/1, getsetsizefromoparg(arg4)/))
    call c_f_pointer(arg6%data, dat3, (/4, getsetsizefromoparg(arg6)/))

    call c_f_pointer(arg0%map_data, map0, (/getmapdimfromoparg(arg0), set%setptr%size/))
    call c_f_pointer(arg2%map_data, map1, (/getmapdimfromoparg(arg2), set%setptr%size/))

    call res_calc_wrapper( &
        name, &
        dat0, &
        dat1, &
        dat2, &
        dat3, &
        map0, &
        map1, &
        set, &
        args, &
        num_dats_indirect, &
        dats_indirect &
    )

    if ((set_size .eq. 0) .or. (set_size .eq. set%setptr%core_size)) then
        call op_mpi_wait_all(size(args), args)
    end if

    call op_mpi_set_dirtybit(size(args), args)
    call op_timers_core(end_time)

    ! todo: review kernel transfer calculation
    transfer = 0.0

    call setkerneltime(3, name // c_null_char, end_time - start_time, transfer, 0.0, 1)
end subroutine

end module

module op2_m_airfoil_3_res_calc_fb

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_3_res_calc_fb

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

subroutine op2_k_airfoil_3_res_calc_wr( &
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

subroutine op2_k_airfoil_3_res_calc_fb( &
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

    call op2_k_airfoil_3_res_calc_wr( &
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

    use op2_m_airfoil_3_res_calc_fb
    use op2_m_airfoil_3_res_calc_m

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
        call op2_k_airfoil_3_res_calc_m( &
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
        call op_check_fallback_mode("airfoil_3_res_calc")
        call op2_k_airfoil_3_res_calc_fb( &
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