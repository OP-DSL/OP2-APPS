PROGRAM jac
  USE op2_kernels
  USE OP2_FORTRAN_DECLARATIONS
  USE, INTRINSIC :: ISO_C_BINDING
  IMPLICIT NONE
  REAL(KIND = 8), PARAMETER :: tolerance = 1E-12
  INTEGER(KIND = 4), PARAMETER :: nn = 6
  INTEGER(KIND = 4), PARAMETER :: niter = 2
  LOGICAL :: valid
  INTEGER(KIND = 4) :: i, j, p
  REAL(KIND = 8) :: u_sum, u_max, alpha, beta
  INTEGER(KIND = 4) :: nnode, nedge
  INTEGER(KIND = 4), DIMENSION(:), ALLOCATABLE :: pp
  REAL(KIND = 8), DIMENSION(:), ALLOCATABLE :: A, r, u, du
  TYPE(op_set) :: nodes, edges
  TYPE(op_map) :: ppedge
  TYPE(op_dat) :: p_A, p_r, p_u, p_du
  CALL op_init(0)
  nnode = (nn - 1) * (nn - 1)
  nedge = nnode + 4 * (nn - 1) * (nn - 2)
  ALLOCATE(pp(nedge * 2))
  ALLOCATE(A(nedge))
  ALLOCATE(r(nnode))
  ALLOCATE(u(nnode))
  ALLOCATE(du(nnode))
  CALL init_data
  CALL op_decl_set(nnode, nodes, "nodes")
  CALL op_decl_set(nedge, edges, "edges")
  CALL op_decl_map(edges, nodes, 2, pp, ppedge, "ppedge")
  CALL op_decl_dat(edges, 1, "real(8)", A, p_A, "p_A")
  CALL op_decl_dat(nodes, 1, "real(8)", r, p_r, "p_r")
  CALL op_decl_dat(nodes, 1, "real(8)", u, p_u, "p_u")
  CALL op_decl_dat(nodes, 1, "real(8)", du, p_du, "p_du")
  DEALLOCATE(pp)
  DEALLOCATE(A)
  DEALLOCATE(r)
  DEALLOCATE(u)
  DEALLOCATE(du)
  alpha = 1.0
  CALL op_decl_const_alpha(alpha, 1)
  CALL op_timing2_start("JAC")
  beta = 1.0
  DO i = 1, niter
    CALL op2_k_jac_1_res("res", edges, op_arg_dat(p_A, - 1, OP_ID, 1, "real(8)", OP_READ), op_arg_dat(p_u, 2, ppedge, 1, "real(8)", OP_READ), op_arg_dat(p_du, 1, ppedge, 1, "real(8)", OP_INC), op_arg_gbl(beta, 1, "real(8)", OP_READ))
    u_sum = 0.0
    u_max = 0.0
    CALL op2_k_jac_2_update("update", nodes, op_arg_dat(p_r, - 1, OP_ID, 1, "real(8)", OP_READ), op_arg_dat(p_du, - 1, OP_ID, 1, "real(8)", OP_RW), op_arg_dat(p_u, - 1, OP_ID, 1, "real(8)", OP_RW), op_arg_gbl(u_sum, 1, "real(8)", OP_INC), op_arg_gbl(u_max, 1, "real(8)", OP_MAX))
    WRITE(*, "(1X, A, F7.4, A, F10.8)") "u max = ", u_max, "; u rms = ", SQRT(u_sum / nnode)
  END DO
  CALL op_timing2_finish
  PRINT *
  CALL op_timing2_output
  ALLOCATE(u(nnode))
  CALL op_fetch_data(p_u, u)
  PRINT *
  WRITE(*, "(1X, A, I0, A)") "Results after ", niter, " iterations:"
  CALL output_data
  valid = check_data()
  IF (valid) THEN
    PRINT *, "Test PASSED"
  ELSE
    PRINT *, "Test FAILED"
  END IF
  DEALLOCATE(u)
  CALL op_exit
  CONTAINS
  SUBROUTINE res(A, u, du, beta)
    IMPLICIT NONE
    REAL(KIND = 8), DIMENSION(1) :: A, u, du, beta
    du(1) = du(1) + beta(1) * A(1) * u(1)
  END SUBROUTINE
  SUBROUTINE update(r, du, u, u_sum, u_max)
    IMPLICIT NONE
    REAL(KIND = 8), DIMENSION(1) :: r, du, u, u_sum, u_max
    u(1) = u(1) + du(1) + alpha * r(1)
    du(1) = 0.0
    u_sum(1) = u_sum(1) + u(1) ** 2
    u_max(1) = MAX(u_max(1), u(1))
  END SUBROUTINE
  SUBROUTINE init_data
    IMPLICIT NONE
    INTEGER(KIND = 4) :: n, e, i2, j2
    INTEGER(KIND = 4), DIMENSION(4) :: i_p, j_p
    i_p = (/- 1, 1, 0, 0/)
    j_p = (/0, 0, - 1, 1/)
    e = 1
    DO i = 1, nn - 1
      DO j = 1, nn - 1
        n = i + (j - 1) * (nn - 1)
        r(n) = 0.0
        u(n) = 0.0
        du(n) = 0.0
        pp(2 * (e - 1) + 1) = n
        pp(2 * (e - 1) + 2) = n
        A(e) = - 1.0
        e = e + 1
        DO p = 1, 4
          i2 = i + i_p(p)
          j2 = j + j_p(p)
          IF (i2 == 0 .OR. i2 == nn .OR. j2 == 0 .OR. j2 == nn) THEN
            r(n) = r(n) + 0.25
          ELSE
            pp(2 * (e - 1) + 1) = n
            pp(2 * (e - 1) + 2) = i2 + (j2 - 1) * (nn - 1)
            A(e) = 0.25
            e = e + 1
          END IF
        END DO
      END DO
    END DO
  END SUBROUTINE
  SUBROUTINE output_data
    IMPLICIT NONE
    DO j = nn - 1, 1, - 1
      DO i = 1, nn - 1
        WRITE(*, "(1X, F7.4)", ADVANCE = "no") u(i + (j - 1) * (nn - 1))
      END DO
      WRITE(*, *)
    END DO
    WRITE(*, *)
  END SUBROUTINE
  FUNCTION check_data() RESULT(valid)
    IMPLICIT NONE
    INTEGER(KIND = 4) :: n
    LOGICAL :: valid
    valid = .TRUE.
    DO i = 1, nn - 1
      DO j = 1, nn - 1
        n = i + (j - 1) * (nn - 1)
        IF ((i == 1 .OR. i == nn - 1) .AND. (j == 1 .OR. j == nn - 1)) THEN
          valid = check_value(u(n), 0.6250_8) .AND. valid
        ELSE IF ((i == 1 .OR. i == nn - 1) .AND. (j == 2 .OR. j == nn - 2)) THEN
          valid = check_value(u(n), 0.4375_8) .AND. valid
        ELSE IF ((j == 1 .OR. j == nn - 1) .AND. (i == 2 .OR. i == nn - 2)) THEN
          valid = check_value(u(n), 0.4375_8) .AND. valid
        ELSE IF ((i == 2 .OR. i == nn - 2) .AND. (j == 2 .OR. j == nn - 2)) THEN
          valid = check_value(u(n), 0.1250_8) .AND. valid
        ELSE IF (i == 1 .OR. i == nn - 1 .OR. j == 1 .OR. j == nn - 1) THEN
          valid = check_value(u(n), 0.3750_8) .AND. valid
        ELSE IF (i == 2 .OR. i == nn - 2 .OR. j == 2 .OR. j == nn - 2) THEN
          valid = check_value(u(n), 0.0625_8) .AND. valid
        ELSE
          valid = check_value(u(n), 0.0000_8) .AND. valid
        END IF
      END DO
    END DO
  END FUNCTION
  FUNCTION check_value(x, ref) RESULT(valid)
    REAL(KIND = 8) :: x, ref
    LOGICAL :: valid
    valid = ABS(x - ref) < tolerance
    IF (.NOT. valid) THEN
      WRITE(*, "(1X, A, F7.4, A, F7.4, A, I0, A, I0)") "Node check failed: expected = ", ref, "; actual = ", x, "; i = ", i, "; j = ", j
    END IF
  END FUNCTION
END PROGRAM