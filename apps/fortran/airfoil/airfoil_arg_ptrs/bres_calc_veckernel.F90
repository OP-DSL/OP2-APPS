!
! auto-generated by op2_fortran.py
!

MODULE BRES_CALC_MODULE
USE OP2_FORTRAN_DECLARATIONS
USE OP2_FORTRAN_RT_SUPPORT
USE ISO_C_BINDING
USE OP2_CONSTANTS


CONTAINS

! user function
SUBROUTINE bres_calc(x1,x2,q1,adt1,res1,bound)
  IMPLICIT NONE
  REAL(kind=8), DIMENSION(2) :: x1
  REAL(kind=8), DIMENSION(2) :: x2
  REAL(kind=8), DIMENSION(4) :: q1
  REAL(kind=8) :: adt1
  REAL(kind=8), DIMENSION(4) :: res1
  INTEGER(kind=4) :: bound
  REAL(kind=8) :: dx,dy,mu,ri,p1,vol1,p2,vol2,f

  dx = x1(1) - x2(1)
  dy = x1(2) - x2(2)
  ri = 1.0 / q1(1)
  p1 = gm1 * (q1(4) - 0.5 * ri * (q1(2) * q1(2) + q1(3) * q1(3)))

  IF (bound .EQ. 1) THEN
    res1(2) = res1(2) + p1 * dy
    res1(3) = res1(3) - p1 * dx
  ELSE
    vol1 = ri * (q1(2) * dy - q1(3) * dx)
    ri = 1.0 / qinf(1)
    p2 = gm1 * (qinf(4) - 0.5 * ri * (qinf(2) * qinf(2) + qinf(3) * qinf(3)))
    vol2 = ri * (qinf(2) * dy - qinf(3) * dx)
    mu = adt1 * eps
    f = 0.5 * (vol1 * q1(1) + vol2 * qinf(1)) + mu * (q1(1) - qinf(1))
    res1(1) = res1(1) + f
    f = 0.5 * (vol1 * q1(2) + p1 * dy + vol2 * qinf(2) + p2 * dy) + mu * (q1(2) - qinf(2))
    res1(2) = res1(2) + f
    f = 0.5 * (vol1 * q1(3) - p1 * dx + vol2 * qinf(3) - p2 * dx) + mu * (q1(3) - qinf(3))
    res1(3) = res1(3) + f
    f = 0.5 * (vol1 * (q1(4) + p1) + vol2 * (qinf(4) + p2)) + mu * (q1(4) - qinf(4))
    res1(4) = res1(4) + f
  END IF
END SUBROUTINE

#define SIMD_VEC 4
#define VECTORIZE
#ifdef VECTORIZE
! user function -- modified for vectorisation
SUBROUTINE bres_calc_vec(x1,x2,q1,adt1,res1,bound,idx)
  !dir$ attributes vector :: bres_calc_vec

  IMPLICIT NONE
  INTEGER(KIND=4) :: idx
  real(8), DIMENSION(SIMD_VEC,2), INTENT(IN) :: x1
  real(8), DIMENSION(SIMD_VEC,2), INTENT(IN) :: x2
  real(8), DIMENSION(SIMD_VEC,4), INTENT(IN) :: q1
  real(8), DIMENSION(SIMD_VEC,1), INTENT(IN) :: adt1
  real(8), DIMENSION(SIMD_VEC,4) :: res1
  INTEGER(kind=4) :: bound
  REAL(kind=8) :: dx
  REAL(kind=8) :: dy
  REAL(kind=8) :: mu
  REAL(kind=8) :: ri
  REAL(kind=8) :: p1
  REAL(kind=8) :: vol1
  REAL(kind=8) :: p2
  REAL(kind=8) :: vol2
  REAL(kind=8) :: f

  dx = x1(idx,1) - x2(idx,1)
  dy = x1(idx,2) - x2(idx,2)
  ri = 1.0 / q1(idx,1)
  p1 = gm1 * (q1(idx,4) - 0.5 * ri * (q1(idx,2) * q1(idx,2) + q1(idx,3) * q1(idx,3)))

  IF (bound .EQ. 1) THEN
    res1(idx,2) = res1(idx,2) + p1 * dy
    res1(idx,3) = res1(idx,3) - p1 * dx
  ELSE
    vol1 = ri * (q1(idx,2) * dy - q1(idx,3) * dx)
    ri = 1.0 / qinf(1)
    p2 = gm1 * (qinf(4) - 0.5 * ri * (qinf(2) * qinf(2) + qinf(3) * qinf(3)))
    vol2 = ri * (qinf(2) * dy - qinf(3) * dx)
    mu = adt1(idx,1) * eps
    f = 0.5 * (vol1 * q1(idx,1) + vol2 * qinf(1)) + mu * (q1(idx,1) - qinf(1))
    res1(idx,1) = res1(idx,1) + f
    f = 0.5 * (vol1 * q1(idx,2) + p1 * dy + vol2 * qinf(2) + p2 * dy) + mu * (q1(idx,2) - qinf(2))
    res1(idx,2) = res1(idx,2) + f
    f = 0.5 * (vol1 * q1(idx,3) - p1 * dx + vol2 * qinf(3) - p2 * dx) + mu * (q1(idx,3) - qinf(3))
    res1(idx,3) = res1(idx,3) + f
    f = 0.5 * (vol1 * (q1(idx,4) + p1) + vol2 * (qinf(4) + p2)) + mu * (q1(idx,4) - qinf(4))
    res1(idx,4) = res1(idx,4) + f
  END IF
end subroutine
#endif

SUBROUTINE op_wrap_bres_calc( &
  &  optflags,        &
  & opDat1Local, &
  & opDat3Local, &
  & opDat4Local, &
  & opDat5Local, &
  & opDat6Local, &
  & opDat1Map, &
  & opDat1MapDim, &
  & opDat3Map, &
  & opDat3MapDim, &
  & bottom,top)
  implicit none
  INTEGER(kind=4), VALUE :: optflags
  real(8) opDat1Local(2,*)
  real(8) opDat3Local(4,*)
  real(8) opDat4Local(1,*)
  real(8) opDat5Local(4,*)
  integer(4) opDat6Local(1,*)
  INTEGER(kind=4) opDat1Map(*)
  INTEGER(kind=4) opDat1MapDim
  INTEGER(kind=4) opDat3Map(*)
  INTEGER(kind=4) opDat3MapDim
  INTEGER(kind=4) bottom,top,i1, i2
  INTEGER(kind=4) map1idx, map2idx, map3idx

  real(8) dat1(SIMD_VEC,2)
  real(8) dat2(SIMD_VEC,2)
  real(8) dat3(SIMD_VEC,4)
  real(8) dat4(SIMD_VEC,1)
  real(8) dat5(SIMD_VEC,4)

  !dir$ attributes align: 64:: dat1
  !dir$ attributes align: 64:: dat2
  !dir$ attributes align: 64:: dat3
  !dir$ attributes align: 64:: dat4
  !dir$ attributes align: 64:: dat5

  !DIR$ ASSUME_ALIGNED opDat1Local : 64
  !DIR$ ASSUME_ALIGNED opDat3Local : 64
  !DIR$ ASSUME_ALIGNED opDat4Local : 64
  !DIR$ ASSUME_ALIGNED opDat5Local : 64
  !DIR$ ASSUME_ALIGNED opDat6Local : 64
  !DIR$ ASSUME_ALIGNED opDat1Map : 64
  !DIR$ ASSUME_ALIGNED opDat3Map : 64
#ifdef VECTORIZE
  DO i1 = bottom, ((top-1)/SIMD_VEC)*SIMD_VEC-1, SIMD_VEC
    !DIR$ SIMD
    DO i2 = 1, SIMD_VEC, 1
      map1idx = opDat1Map(1 + (i1+i2-1) * opDat1MapDim + 0) + 1
      map2idx = opDat1Map(1 + (i1+i2-1) * opDat1MapDim + 1) + 1
      map3idx = opDat3Map(1 + (i1+i2-1) * opDat3MapDim + 0) + 1


      IF (BTEST(optflags,0)) THEN
        dat1(i2,1) = opDat1Local(1,map1idx)
        dat1(i2,2) = opDat1Local(2,map1idx)
      END IF

      dat2(i2,1) = opDat1Local(1,map2idx)
      dat2(i2,2) = opDat1Local(2,map2idx)

      dat3(i2,1) = opDat3Local(1,map3idx)
      dat3(i2,2) = opDat3Local(2,map3idx)
      dat3(i2,3) = opDat3Local(3,map3idx)
      dat3(i2,4) = opDat3Local(4,map3idx)

      dat4(i2,1) = opDat4Local(1,map3idx)

      dat5(i2,:) = 0.0
    END DO
    !DIR$ SIMD
    !DIR$ FORCEINLINE
    DO i2 = 1, SIMD_VEC, 1
      ! vectorized kernel call
      CALL bres_calc_vec( &
      & dat1, &
      & dat2, &
      & dat3, &
      & dat4, &
      & dat5, &
      & opDat6Local(1,(i1+i2-1)+1), &
      & i2)
    END DO
    DO i2 = 1, SIMD_VEC, 1
      map3idx = opDat3Map(1 + (i1+i2-1) * opDat3MapDim + 0) + 1

      IF (BTEST(optflags,0)) THEN
      END IF
      opDat5Local(1,map3idx) = opDat5Local(1,map3idx) + dat5(i2,1)
      opDat5Local(2,map3idx) = opDat5Local(2,map3idx) + dat5(i2,2)
      opDat5Local(3,map3idx) = opDat5Local(3,map3idx) + dat5(i2,3)
      opDat5Local(4,map3idx) = opDat5Local(4,map3idx) + dat5(i2,4)

    END DO
  END DO
  ! remainder
  DO i1 = ((top-1)/SIMD_VEC)*SIMD_VEC, top-1, 1
#else
  !DIR$ FORCEINLINE
  DO i1 = bottom, top-1, 1
#endif
    map1idx = opDat1Map(1 + i1 * opDat1MapDim + 0)+1
    map2idx = opDat1Map(1 + i1 * opDat1MapDim + 1)+1
    map3idx = opDat3Map(1 + i1 * opDat3MapDim + 0)+1
    ! kernel call
    CALL bres_calc( &
    & opDat1Local(1,map1idx), &
    & opDat1Local(1,map2idx), &
    & opDat3Local(1,map3idx), &
    & opDat4Local(1,map3idx), &
    & opDat5Local(1,map3idx), &
    & opDat6Local(1,i1+1) &
    & )
  END DO
END SUBROUTINE
SUBROUTINE bres_calc_host( userSubroutine, set, &
  & opArg1, &
  & opArg2, &
  & opArg3, &
  & opArg4, &
  & opArg5, &
  & opArg6 )

  IMPLICIT NONE
  character(kind=c_char,len=*), INTENT(IN) :: userSubroutine
  type ( op_set ) , INTENT(IN) :: set

  type ( op_arg ) , INTENT(IN) :: opArg1
  type ( op_arg ) , INTENT(IN) :: opArg2
  type ( op_arg ) , INTENT(IN) :: opArg3
  type ( op_arg ) , INTENT(IN) :: opArg4
  type ( op_arg ) , INTENT(IN) :: opArg5
  type ( op_arg ) , INTENT(IN) :: opArg6

  type ( op_arg ) , DIMENSION(6) :: opArgArray
  INTEGER(kind=4) :: numberOfOpDats
  REAL(kind=4) :: dataTransfer
  INTEGER(kind=4), DIMENSION(1:8) :: timeArrayStart
  INTEGER(kind=4), DIMENSION(1:8) :: timeArrayEnd
  REAL(kind=8) :: startTime
  REAL(kind=8) :: endTime
  INTEGER(kind=4) :: returnSetKernelTiming
  INTEGER(kind=4) :: n_upper
  type ( op_set_core ) , POINTER :: opSetCore

  INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat1Map
  INTEGER(kind=4) :: opDat1MapDim
  real(8), POINTER, DIMENSION(:) :: opDat1Local
  INTEGER(kind=4) :: opDat1Cardinality

  INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat3Map
  INTEGER(kind=4) :: opDat3MapDim
  real(8), POINTER, DIMENSION(:) :: opDat3Local
  INTEGER(kind=4) :: opDat3Cardinality

  INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat4Map
  INTEGER(kind=4) :: opDat4MapDim
  real(8), POINTER, DIMENSION(:) :: opDat4Local
  INTEGER(kind=4) :: opDat4Cardinality

  INTEGER(kind=4), POINTER, DIMENSION(:) :: opDat5Map
  INTEGER(kind=4) :: opDat5MapDim
  real(8), POINTER, DIMENSION(:) :: opDat5Local
  INTEGER(kind=4) :: opDat5Cardinality

  integer(4), POINTER, DIMENSION(:) :: opDat6Local
  INTEGER(kind=4) :: opDat6Cardinality

  real(8), POINTER, DIMENSION(:) :: opDat1OptPtr

  INTEGER(kind=4) :: i1
  INTEGER(kind=4) :: optflags
  optflags = 0
  IF (opArg1%opt == 1) THEN
    optflags = IBSET(optflags,0)
  END IF

  numberOfOpDats = 6

  opArgArray(1) = opArg1
  opArgArray(2) = opArg2
  opArgArray(3) = opArg3
  opArgArray(4) = opArg4
  opArgArray(5) = opArg5
  opArgArray(6) = opArg6

  returnSetKernelTiming = setKernelTime(3 , userSubroutine//C_NULL_CHAR, &
  & 0.d0, 0.00000,0.00000, 0)
  call op_timers_core(startTime)

  n_upper = op_mpi_halo_exchanges(set%setCPtr,numberOfOpDats,opArgArray)

  opSetCore => set%setPtr

  opDat1Cardinality = opArg1%dim * getSetSizeFromOpArg(opArg1)
  opDat1MapDim = getMapDimFromOpArg(opArg1)
  opDat3Cardinality = opArg3%dim * getSetSizeFromOpArg(opArg3)
  opDat3MapDim = getMapDimFromOpArg(opArg3)
  opDat4Cardinality = opArg4%dim * getSetSizeFromOpArg(opArg4)
  opDat4MapDim = getMapDimFromOpArg(opArg4)
  opDat5Cardinality = opArg5%dim * getSetSizeFromOpArg(opArg5)
  opDat5MapDim = getMapDimFromOpArg(opArg5)
  opDat6Cardinality = opArg6%dim * getSetSizeFromOpArg(opArg6)
  CALL c_f_pointer(opArg1%data,opDat1Local,(/opDat1Cardinality/))
  CALL c_f_pointer(opArg1%map_data,opDat1Map,(/opSetCore%size*opDat1MapDim/))
  CALL c_f_pointer(opArg3%data,opDat3Local,(/opDat3Cardinality/))
  CALL c_f_pointer(opArg3%map_data,opDat3Map,(/opSetCore%size*opDat3MapDim/))
  CALL c_f_pointer(opArg4%data,opDat4Local,(/opDat4Cardinality/))
  CALL c_f_pointer(opArg4%map_data,opDat4Map,(/opSetCore%size*opDat4MapDim/))
  CALL c_f_pointer(opArg5%data,opDat5Local,(/opDat5Cardinality/))
  CALL c_f_pointer(opArg5%map_data,opDat5Map,(/opSetCore%size*opDat5MapDim/))
  CALL c_f_pointer(opArg6%data,opDat6Local,(/opDat6Cardinality/))


  CALL op_mpi_wait_all(numberOfOpDats,opArgArray)
  CALL op_wrap_bres_calc( &
  & optflags, &
  & opDat1Local, &
  & opDat3Local, &
  & opDat4Local, &
  & opDat5Local, &
  & opDat6Local, &
  & opDat1Map, &
  & opDat1MapDim, &
  & opDat3Map, &
  & opDat3MapDim, &
  & 0, n_upper)

  CALL op_mpi_set_dirtybit(numberOfOpDats,opArgArray)

  call op_timers_core(endTime)

  dataTransfer = 0.0
  IF (opArg1%opt == 1) THEN
    dataTransfer = dataTransfer + opArg1%size * MIN(n_upper,getSetSizeFromOpArg(opArg1))
  END IF
  dataTransfer = dataTransfer + opArg3%size * MIN(n_upper,getSetSizeFromOpArg(opArg3))
  dataTransfer = dataTransfer + opArg4%size * MIN(n_upper,getSetSizeFromOpArg(opArg4))
  dataTransfer = dataTransfer + opArg5%size * MIN(n_upper,getSetSizeFromOpArg(opArg5)) * 2.d0
  dataTransfer = dataTransfer + opArg6%size * MIN(n_upper,getSetSizeFromOpArg(opArg6))
  dataTransfer = dataTransfer + n_upper * opDat1MapDim * 4.d0
  dataTransfer = dataTransfer + n_upper * opDat3MapDim * 4.d0
  returnSetKernelTiming = setKernelTime(3 , userSubroutine//C_NULL_CHAR, &
  & endTime-startTime, dataTransfer, 0.00000_4, 1)

END SUBROUTINE
END MODULE