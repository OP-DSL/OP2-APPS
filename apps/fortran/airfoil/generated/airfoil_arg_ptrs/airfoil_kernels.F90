MODULE airfoil_kernels
  USE airfoil_constants
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: save_soln, adt_calc, res_calc, bres_calc, update
  CONTAINS
  SUBROUTINE save_soln(q, qold)
    REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
    REAL(KIND = 8), DIMENSION(4), INTENT(OUT) :: qold
    INTEGER(KIND = 4) :: i
    DO i = 1, 4
      qold(i) = q(i)
    END DO
  END SUBROUTINE
  SUBROUTINE adt_calc(x1, x2, x3, x4, q, adt)
    REAL(KIND = 8), DIMENSION(2), INTENT(IN) :: x1, x2, x3, x4
    REAL(KIND = 8), DIMENSION(4), INTENT(IN) :: q
    REAL(KIND = 8), INTENT(OUT) :: adt
    REAL(KIND = 8) :: dx, dy, ri, u, v, c
    ri = 1.0_8 / q(1)
    u = ri * q(2)
    v = ri * q(3)
    c = SQRT(gam * gm1 * (ri * q(4) - 0.5_8 * (u ** 2 + v ** 2)))
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
    adt = adt / cfl
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
    p1 = gm1 * (q1(4) - 0.5_8 * ri * (q1(2) ** 2 + q1(3) ** 2))
    vol1 = ri * (q1(2) * dy - q1(3) * dx)
    ri = 1.0_8 / q2(1)
    p2 = gm1 * (q2(4) - 0.5_8 * ri * (q2(2) ** 2 + q2(3) ** 2))
    vol2 = ri * (q2(2) * dy - q2(3) * dx)
    mu = 0.5_8 * (adt1 + adt2) * eps
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
    p1 = gm1 * (q1(4) - 0.5_8 * ri * (q1(2) ** 2 + q1(3) ** 2))
    IF (bound == 1) THEN
      res1(2) = res1(2) + p1 * dy
      res1(3) = res1(3) - p1 * dx
      RETURN
    END IF
    vol1 = ri * (q1(2) * dy - q1(3) * dx)
    ri = 1.0_8 / qinf(1)
    p2 = gm1 * (qinf(4) - 0.5_8 * ri * (qinf(2) ** 2 + qinf(3) ** 2))
    vol2 = ri * (qinf(2) * dy - qinf(3) * dx)
    mu = adt1 * eps
    f = 0.5_8 * (vol1 * q1(1) + vol2 * qinf(1)) + mu * (q1(1) - qinf(1))
    res1(1) = res1(1) + f
    f = 0.5_8 * (vol1 * q1(2) + p1 * dy + vol2 * qinf(2) + p2 * dy) + mu * (q1(2) - qinf(2))
    res1(2) = res1(2) + f
    f = 0.5_8 * (vol1 * q1(3) - p1 * dx + vol2 * qinf(3) - p2 * dx) + mu * (q1(3) - qinf(3))
    res1(3) = res1(3) + f
    f = 0.5_8 * (vol1 * (q1(4) + p1) + vol2 * (qinf(4) + p2)) + mu * (q1(4) - qinf(4))
    res1(4) = res1(4) + f
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
END MODULE