#define op2_s(idx, stride) 1 + ((idx) - 1) * op2_stride_##stride##_d

module op2_m_airfoil_4_bres_calc_main

    use cudafor
    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support
    use cudaconfigurationparams

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_4_bres_calc_main

contains

attributes(device) &
SUBROUTINE bres_calc(x1, x2, q1, adt1, res1, bound)
  REAL(KIND = 8), DIMENSION(2), INTENT(IN) :: x1, x2
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q1
  REAL(KIND = 8), INTENT(IN) :: adt1
  REAL(KIND = 8), DIMENSION(4), INTENT(INOUT) :: res1
  INTEGER(KIND = 4), INTENT(IN) :: bound
  REAL(KIND = 8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f
  INTEGER(KIND = 4) :: op2_ret
  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)
  ri = 1.0_8 / q1(1)
  p1 = op2_const_gm1_d * (q1(4) - 0.5_8 * ri * (q1(2) ** 2 + q1(3) ** 2))
  IF (bound == 1) THEN
    op2_ret = atomicAdd(res1(2), 0.0D0 + p1 * dy)
    op2_ret = atomicAdd(res1(3), 0.0D0 - p1 * dx)
    RETURN
  END IF
  vol1 = ri * (q1(2) * dy - q1(3) * dx)
  ri = 1.0_8 / op2_const_qinf_d(1)
  p2 = op2_const_gm1_d * (op2_const_qinf_d(4) - 0.5_8 * ri * (op2_const_qinf_d(2) ** 2 + op2_const_qinf_d(3) ** 2))
  vol2 = ri * (op2_const_qinf_d(2) * dy - op2_const_qinf_d(3) * dx)
  mu = adt1 * op2_const_eps_d
  f = 0.5_8 * (vol1 * q1(1) + vol2 * op2_const_qinf_d(1)) + mu * (q1(1) - op2_const_qinf_d(1))
  op2_ret = atomicAdd(res1(1), 0.0D0 + f)
  f = 0.5_8 * (vol1 * q1(2) + p1 * dy + vol2 * op2_const_qinf_d(2) + p2 * dy) + mu * (q1(2) - op2_const_qinf_d(2))
  op2_ret = atomicAdd(res1(2), 0.0D0 + f)
  f = 0.5_8 * (vol1 * q1(3) - p1 * dx + vol2 * op2_const_qinf_d(3) - p2 * dx) + mu * (q1(3) - op2_const_qinf_d(3))
  op2_ret = atomicAdd(res1(3), 0.0D0 + f)
  f = 0.5_8 * (vol1 * (q1(4) + p1) + vol2 * (op2_const_qinf_d(4) + p2)) + mu * (q1(4) - op2_const_qinf_d(4))
  op2_ret = atomicAdd(res1(4), 0.0D0 + f)
END SUBROUTINE

attributes(global) &
subroutine op2_k_airfoil_4_bres_calc_wrapper( &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    dat4, &
    map0, &
    map1, &
    start, &
    end, &
    set_size &
)
    implicit none

    ! parameters
    real(8), dimension(*) :: dat0
    real(8), dimension(*) :: dat1
    real(8), dimension(*) :: dat2
    real(8), dimension(*) :: dat3
    integer(4), dimension(*) :: dat4

    integer(4), dimension(*) :: map0
    integer(4), dimension(*) :: map1

    integer(4), value :: start, end, set_size

    ! locals
    integer(4) :: thread_id, d, n, ret

    thread_id = threadIdx%x + (blockIdx%x - 1) * blockDim%x

    do n = thread_id + start, end, blockDim%x * gridDim%x
        call bres_calc( &
            dat0(map0(0 * set_size + n) * 2 + 1), &
            dat0(map0(1 * set_size + n) * 2 + 1), &
            dat1(map1(0 * set_size + n) * 4 + 1), &
            dat2(map1(0 * set_size + n) * 1 + 1), &
            dat3(map1(0 * set_size + n) * 4 + 1), &
            dat4((n - 1) * 1 + 1) &
        )
    end do
end subroutine

subroutine op2_k_airfoil_4_bres_calc_main( &
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

    integer(4) :: n_exec, col, block, round, dim, err, d

    real(8), dimension(:), pointer, device :: dat0_d
    real(8), dimension(:), pointer, device :: dat1_d
    real(8), dimension(:), pointer, device :: dat2_d
    real(8), dimension(:), pointer, device :: dat3_d
    integer(4), dimension(:), pointer, device :: dat4_d

    integer(4), dimension(:, :), pointer, device :: map0_d
    integer(4), dimension(:, :), pointer, device :: map1_d

    real(8) :: start_time, end_time
    real(4) :: transfer

    integer(4) :: num_blocks, max_blocks, block_size, block_limit
    integer(4) :: start, end
    integer(4), dimension(4) :: sections

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4
    args(6) = arg5

    call op_timing2_enter_kernel("airfoil_4_bres_calc", "CUDA", "Indirect (atomics)")
    call op_timing2_enter("Init")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges_grouped(set%setcptr, size(args), args, 2)

    if (n_exec == 0) then
        call op_timing2_exit()
        call op_timing2_exit()

        call op_mpi_wait_all_grouped(size(args), args, 2)
        call op_mpi_set_dirtybit_cuda(size(args), args)
        err = cudaDeviceSynchronize()

        if (err /= 0) then
            print *, cudaGetErrorString(err)
        end if

        call op_timing2_exit()
        return
    end if

    call op_timing2_next("Update consts")
    call op_update_const_cuda_qinf()
    call op_update_const_cuda_eps()
    call op_update_const_cuda_gm1()

    call op_timing2_exit()

    call setGblIncAtomic(logical(.false., c_bool))
    block_size = getBlockSize(name // c_null_char, set%setptr%size)
    block_limit = getBlockLimit(args, size(args), block_size, name // c_null_char)

    max_blocks = (max(set%setptr%core_size, &
        set%setptr%size + set%setptr%exec_size - set%setptr%core_size) - 1 + (block_size - 1)) / block_size
    max_blocks = min(max_blocks, block_limit)

    call op_timing2_enter("Prepare GBLs")
    call prepareDeviceGbls(args, size(args), block_size * max_blocks)
    call op_timing2_exit()

    arg0 = args(1)
    arg1 = args(2)
    arg2 = args(3)
    arg3 = args(4)
    arg4 = args(5)
    arg5 = args(6)

    call c_f_pointer(arg0%data_d, dat0_d, (/2 * getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg2%data_d, dat1_d, (/4 * getsetsizefromoparg(arg2)/))
    call c_f_pointer(arg3%data_d, dat2_d, (/1 * getsetsizefromoparg(arg3)/))
    call c_f_pointer(arg4%data_d, dat3_d, (/4 * getsetsizefromoparg(arg4)/))
    call c_f_pointer(arg5%data_d, dat4_d, (/1 * getsetsizefromoparg(arg5)/))

    call c_f_pointer(arg0%map_data_d, map0_d, (/set%setptr%size, getmapdimfromoparg(arg0)/))
    call c_f_pointer(arg2%map_data_d, map1_d, (/set%setptr%size, getmapdimfromoparg(arg2)/))

    call op_timing2_next("Computation")
    sections = (/0, set%setptr%core_size, set%setptr%size + set%setptr%exec_size, 0/)

    call op_timing2_enter("Kernel")
    do round = 1, 2
        if (round == 2) then
            call op_timing2_next("MPI Wait")
            call op_mpi_wait_all_grouped(size(args), args, 2)
            call op_timing2_next("Kernel")
        end if

        start = sections(round)
        end = sections(round + 1)

        if (end - start > 0) then
            num_blocks = (end - start + (block_size - 1)) / block_size
            num_blocks = min(num_blocks, block_limit)

            call op2_k_airfoil_4_bres_calc_wrapper<<<num_blocks, block_size>>>( &
                dat0_d, &
                dat1_d, &
                dat2_d, &
                dat3_d, &
                dat4_d, &
                map0_d, &
                map1_d, &
                start, &
                end, &
                set%setptr%size + set%setptr%exec_size &
            )
        end if
    end do

    call op_timing2_exit()
    call op_timing2_exit()

    call op_timing2_enter("Finalise")
    call op_mpi_set_dirtybit_cuda(size(args), args)

    err = cudaDeviceSynchronize()

    if (err /= 0) then
        print *, cudaGetErrorString(err)
    end if

    call op_timing2_exit()
    call op_timing2_exit()
end subroutine

end module

module op2_m_airfoil_4_bres_calc_fallback

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    implicit none

    private
    public :: op2_k_airfoil_4_bres_calc_fallback

contains

SUBROUTINE bres_calc(x1, x2, q1, adt1, res1, bound)
  REAL(KIND = 8), DIMENSION(2), INTENT(IN) :: x1, x2
  REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q1
  REAL(KIND = 8), INTENT(IN) :: adt1
  REAL(KIND = 8), DIMENSION(4), INTENT(INOUT) :: res1
  INTEGER(KIND = 4), INTENT(IN) :: bound
  REAL(KIND = 8) :: dx, dy, mu, ri, p1, vol1, p2, vol2, f
  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)
  ri = 1.0_8 / q1(1)
  p1 = op2_const_gm1 * (q1(4) - 0.5_8 * ri * (q1(2) ** 2 + q1(3) ** 2))
  IF (bound == 1) THEN
    res1(2) = res1(2) + p1 * dy
    res1(3) = res1(3) - p1 * dx
    RETURN
  END IF
  vol1 = ri * (q1(2) * dy - q1(3) * dx)
  ri = 1.0_8 / op2_const_qinf(1)
  p2 = op2_const_gm1 * (op2_const_qinf(4) - 0.5_8 * ri * (op2_const_qinf(2) ** 2 + op2_const_qinf(3) ** 2))
  vol2 = ri * (op2_const_qinf(2) * dy - op2_const_qinf(3) * dx)
  mu = adt1 * op2_const_eps
  f = 0.5_8 * (vol1 * q1(1) + vol2 * op2_const_qinf(1)) + mu * (q1(1) - op2_const_qinf(1))
  res1(1) = res1(1) + f
  f = 0.5_8 * (vol1 * q1(2) + p1 * dy + vol2 * op2_const_qinf(2) + p2 * dy) + mu * (q1(2) - op2_const_qinf(2))
  res1(2) = res1(2) + f
  f = 0.5_8 * (vol1 * q1(3) - p1 * dx + vol2 * op2_const_qinf(3) - p2 * dx) + mu * (q1(3) - op2_const_qinf(3))
  res1(3) = res1(3) + f
  f = 0.5_8 * (vol1 * (q1(4) + p1) + vol2 * (op2_const_qinf(4) + p2)) + mu * (q1(4) - op2_const_qinf(4))
  res1(4) = res1(4) + f
END SUBROUTINE

subroutine op2_k_airfoil_4_bres_calc_wrapper( &
    dat0, &
    dat1, &
    dat2, &
    dat3, &
    dat4, &
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
    integer(4), dimension(:, :) :: dat4

    integer(4), dimension(:, :) :: map0
    integer(4), dimension(:, :) :: map1

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

        call bres_calc( &
            dat0(:, map0(1, n) + 1), &
            dat0(:, map0(2, n) + 1), &
            dat1(:, map1(1, n) + 1), &
            dat2(1, map1(1, n) + 1), &
            dat3(:, map1(1, n) + 1), &
            dat4(1, n) &
        )
    end do
end subroutine

subroutine op2_k_airfoil_4_bres_calc_fallback( &
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
    real(8), pointer, dimension(:, :) :: dat3
    integer(4), pointer, dimension(:, :) :: dat4

    integer(4), pointer, dimension(:, :) :: map0
    integer(4), pointer, dimension(:, :) :: map1

    real(4) :: transfer

    args(1) = arg0
    args(2) = arg1
    args(3) = arg2
    args(4) = arg3
    args(5) = arg4
    args(6) = arg5

    call op_timing2_enter_kernel("airfoil_4_bres_calc", "seq", "Indirect")

    call op_timing2_enter("MPI Exchanges")
    n_exec = op_mpi_halo_exchanges(set%setcptr, size(args), args)

    call op_timing2_next("Computation")

    call c_f_pointer(arg0%data, dat0, (/2, getsetsizefromoparg(arg0)/))
    call c_f_pointer(arg2%data, dat1, (/4, getsetsizefromoparg(arg2)/))
    call c_f_pointer(arg3%data, dat2, (/1, getsetsizefromoparg(arg3)/))
    call c_f_pointer(arg4%data, dat3, (/4, getsetsizefromoparg(arg4)/))
    call c_f_pointer(arg5%data, dat4, (/1, getsetsizefromoparg(arg5)/))

    call c_f_pointer(arg0%map_data, map0, (/getmapdimfromoparg(arg0), set%setptr%size/))
    call c_f_pointer(arg2%map_data, map1, (/getmapdimfromoparg(arg2), set%setptr%size/))

    call op2_k_airfoil_4_bres_calc_wrapper( &
        dat0, &
        dat1, &
        dat2, &
        dat3, &
        dat4, &
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

module op2_m_airfoil_4_bres_calc

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_m_airfoil_4_bres_calc_fallback
    use op2_m_airfoil_4_bres_calc_main

    implicit none

    private
    public :: op2_k_airfoil_4_bres_calc

contains

subroutine op2_k_airfoil_4_bres_calc( &
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

    if (op_check_whitelist("airfoil_4_bres_calc")) then
        call op2_k_airfoil_4_bres_calc_main( &
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
        call op2_k_airfoil_4_bres_calc_fallback( &
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