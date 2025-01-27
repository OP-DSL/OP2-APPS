MODULE airfoil_constants
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: gam, gm1, cfl, eps, mach, alpha, qinf
  REAL(KIND = 8), PARAMETER :: gam = 1.4_8
  REAL(KIND = 8), PARAMETER :: gm1 = gam - 1.0_8
  REAL(KIND = 8), PARAMETER :: cfl = 0.9_8
  REAL(KIND = 8), PARAMETER :: eps = 0.05_8
  REAL(KIND = 8), PARAMETER :: mach = 0.4_8
  REAL(KIND = 8), PARAMETER :: alpha = 3.0_8 * ATAN(1.0_8) / 45.0_8
  REAL(KIND = 8), PARAMETER :: p = 1.0_8
  REAL(KIND = 8), PARAMETER :: r = 1.0_8
  REAL(KIND = 8), PARAMETER :: u = SQRT(gam * p / r) * mach
  REAL(KIND = 8), PARAMETER :: e = p / (r * gm1) + 0.5_8 * u ** 2
  REAL(KIND = 8), DIMENSION(4), PARAMETER :: qinf = (/r, r * u, 0.0_8, r * e/)
END MODULE