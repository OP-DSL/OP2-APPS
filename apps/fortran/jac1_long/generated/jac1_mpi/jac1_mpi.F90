PROGRAM jac_distributed
  USE op2_kernels
  USE OP2_fortran_rt_support
  USE mpi
  IMPLICIT NONE
  INTEGER, PARAMETER :: idx_k = SELECTED_INT_KIND(18)
  INTEGER(KIND = idx_k), PARAMETER :: nn = INT(2 ** 16, kind = idx_k)
  INTEGER, PARAMETER :: niter = 2
  REAL(KIND = 8), PARAMETER :: tolerance = 1.0E-12_8
  REAL(KIND = 8) :: alpha
  INTEGER :: my_rank, comm_size, ierr
  INTEGER(KIND = idx_k) :: nnode
  INTEGER(KIND = idx_k) :: nedge
  INTEGER(KIND = idx_k) :: node_start
  INTEGER(KIND = idx_k) :: g_nnode
  INTEGER(KIND = idx_k), DIMENSION(:), ALLOCATABLE :: pp
  REAL(KIND = 8), DIMENSION(:), ALLOCATABLE :: A
  REAL(KIND = 8), DIMENSION(:), ALLOCATABLE :: r
  REAL(KIND = 8), DIMENSION(:), ALLOCATABLE :: u
  REAL(KIND = 8), DIMENSION(:), ALLOCATABLE :: du
  TYPE(op_set) :: nodes, edges
  TYPE(op_map) :: ppedge
  TYPE(op_dat) :: p_A, p_r, p_u, p_du
  INTEGER :: iter
  REAL(KIND = 8) :: u_sum, u_max, beta
  INTEGER :: validation_result
  INTEGER(KIND = idx_k) :: nedge_local, edge_counter
  INTEGER(KIND = idx_k) :: local_idx, global_idx, i, j, i2, j2, neighbor_global_idx
  INTEGER :: pass
  CALL MPI_Init(ierr)
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierr)
  CALL MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  CALL op_init(2)
  g_nnode = (nn - 1) * (nn - 1)
  IF (my_rank == 0) THEN
    PRINT *, "Global number of nodes = ", g_nnode
  END IF
  nnode = compute_local_size(g_nnode, INT(comm_size, idx_k), INT(my_rank, idx_k))
  node_start = compute_local_offset(g_nnode, INT(comm_size, idx_k), INT(my_rank, idx_k))
  nedge_local = 0_idx_k
  DO local_idx = 1_idx_k, nnode
    global_idx = node_start + local_idx - 1_idx_k
    j = global_idx / (nn - 1) + 1_idx_k
    i = MOD(global_idx, nn - 1) + 1_idx_k
    nedge_local = nedge_local + 1_idx_k
    DO pass = 0, 3
      i2 = i
      j2 = j
      IF (pass == 0) i2 = i2 - 1_idx_k
      IF (pass == 1) i2 = i2 + 1_idx_k
      IF (pass == 2) j2 = j2 - 1_idx_k
      IF (pass == 3) j2 = j2 + 1_idx_k
      IF (i2 > 0_idx_k .AND. i2 < nn .AND. j2 > 0_idx_k .AND. j2 < nn) THEN
        nedge_local = nedge_local + 1_idx_k
      END IF
    END DO
  END DO
  nedge = nedge_local
  ALLOCATE(pp(2 * nedge), STAT = ierr)
  IF (ierr /= 0) STOP 'Allocation failed for pp'
  ALLOCATE(a(nedge), STAT = ierr)
  IF (ierr /= 0) STOP 'Allocation failed for A'
  ALLOCATE(r(nnode), STAT = ierr)
  IF (ierr /= 0) STOP 'Allocation failed for r'
  ALLOCATE(u(nnode), STAT = ierr)
  IF (ierr /= 0) STOP 'Allocation failed for u'
  ALLOCATE(du(nnode), STAT = ierr)
  IF (ierr /= 0) STOP 'Allocation failed for du'
  edge_counter = 0_idx_k
  DO local_idx = 1_idx_k, nnode
    global_idx = node_start + local_idx - 1_idx_k
    j = global_idx / (nn - 1) + 1_idx_k
    i = MOD(global_idx, nn - 1) + 1_idx_k
    r(local_idx) = 0.0_8
    u(local_idx) = 0.0_8
    du(local_idx) = 0.0_8
    edge_counter = edge_counter + 1_idx_k
    pp(2 * edge_counter - 1) = global_idx
    pp(2 * edge_counter) = global_idx
    A(edge_counter) = - 1.0_8
    DO pass = 0, 3
      i2 = i
      j2 = j
      IF (pass == 0) i2 = i2 - 1_idx_k
      IF (pass == 1) i2 = i2 + 1_idx_k
      IF (pass == 2) j2 = j2 - 1_idx_k
      IF (pass == 3) j2 = j2 + 1_idx_k
      IF (i2 == 0_idx_k .OR. i2 == nn .OR. j2 == 0_idx_k .OR. j2 == nn) THEN
        r(local_idx) = r(local_idx) + 0.25_8
      ELSE
        neighbor_global_idx = (i2 - 1) + (j2 - 1) * (nn - 1)
        edge_counter = edge_counter + 1_idx_k
        pp(2 * edge_counter - 1) = global_idx
        pp(2 * edge_counter) = neighbor_global_idx
        A(edge_counter) = 0.25_8
      END IF
    END DO
  END DO
  DO local_idx = 1_idx_k, nedge * 2
    pp(local_idx) = pp(local_idx) + 1
  END DO
  IF (edge_counter /= nedge) THEN
    PRINT *, "Rank ", my_rank, ": Mismatch in edge count! Calculated=", nedge, " Filled=", edge_counter
    CALL MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  END IF
  CALL op_decl_set(INT(nnode, 4), nodes, "nodes")
  CALL op_decl_set(INT(nedge, 4), edges, "edges")
  CALL op_decl_map_long(edges, nodes, 2, pp, ppedge, "ppedge")
  CALL op_decl_dat(edges, 1, "real(8)", A, p_A, "p_A")
  CALL op_decl_dat(nodes, 1, "real(8)", r, p_r, "p_r")
  CALL op_decl_dat(nodes, 1, "real(8)", u, p_u, "p_u")
  CALL op_decl_dat(nodes, 1, "real(8)", du, p_du, "p_du")
  DEALLOCATE(A)
  DEALLOCATE(r)
  DEALLOCATE(du)
  alpha = 1.0_8
  CALL op_decl_const_alpha(alpha, 1)
  CALL op_partition("PARMETIS", "KWAY", edges, ppedge, p_u)
  CALL op_profile_start("Jacobi")
  CALL op_profile_enter("Main computation")
  beta = 1.0_8
  DO iter = 1, niter
    CALL op2_k_jac1_mpi_1_res_kernel("res_kernel", edges, op_arg_dat(p_A, - 1, OP_ID, 1, "real(8)", OP_READ), op_arg_dat(p_u, 2, ppedge, 1, "real(8)", OP_READ), op_arg_dat(p_du, 1, ppedge, 1, "real(8)", OP_INC), op_arg_gbl(beta, 1, "real(8)", OP_READ))
    u_sum = 0.0_8
    u_max = 0.0_8
    CALL op2_k_jac1_mpi_2_update_kernel("update_kernel", nodes, op_arg_dat(p_r, - 1, OP_ID, 1, "real(8)", OP_READ), op_arg_dat(p_du, - 1, OP_ID, 1, "real(8)", OP_RW), op_arg_dat(p_u, - 1, OP_ID, 1, "real(8)", OP_INC), op_arg_gbl(u_sum, 1, "real(8)", OP_INC), op_arg_gbl(u_max, 1, "real(8)", OP_MAX))
    IF (my_rank == 0) THEN
      WRITE(*, "(4X, I0, E16.7, 4X, A, E16.7)") iter, u_max, "u rms = ", SQRT(u_sum / DBLE(g_nnode))
    END IF
  END DO
  CALL op_profile_end
  CALL op_profile_output
  IF (.NOT. ALLOCATED(u)) ALLOCATE(u(nnode), STAT = ierr)
  IF (ierr /= 0) STOP 'Allocation failed for u (fetch)'
  CALL op_fetch_data(p_u, u)
  validation_result = distributed_check_result(u, nn, node_start, nnode, tolerance, my_rank)
  CALL MPI_Barrier(MPI_COMM_WORLD, ierr)
  CALL op_exit
  IF (ALLOCATED(u)) DEALLOCATE(u)
  IF (ALLOCATED(pp)) DEALLOCATE(pp)
  IF (validation_result /= 0 .AND. my_rank == 0) THEN
    PRINT *, "Exiting with status 1 due to validation failure."
    CALL exit(1)
  ELSE IF (my_rank == 0) THEN
    PRINT *, "Exiting with status 0 (success)."
  END IF
  CONTAINS
  SUBROUTINE res_kernel(A, u, du, beta)
    IMPLICIT NONE
    REAL(KIND = 8), INTENT(IN) :: A
    REAL(KIND = 8), INTENT(IN) :: u
    REAL(KIND = 8), INTENT(INOUT) :: du
    REAL(KIND = 8), INTENT(IN) :: beta
    du = du + beta * A * u
  END SUBROUTINE res_kernel
  SUBROUTINE update_kernel(r, du, u, u_sum, u_max)
    IMPLICIT NONE
    REAL(KIND = 8), INTENT(IN) :: r
    REAL(KIND = 8), INTENT(INOUT) :: du
    REAL(KIND = 8), INTENT(INOUT) :: u
    REAL(KIND = 8), INTENT(INOUT) :: u_sum
    REAL(KIND = 8), INTENT(INOUT) :: u_max
    u = u + du + alpha * r
    du = 0.0_8
    u_sum = u_sum + u ** 2
    u_max = MAX(u_max, u)
  END SUBROUTINE update_kernel
  FUNCTION compute_local_size(global_size, mpi_comm_size, mpi_rank) RESULT(local_size)
    IMPLICIT NONE
    INTEGER(KIND = idx_k), INTENT(IN) :: global_size, mpi_comm_size, mpi_rank
    INTEGER(KIND = idx_k) :: local_size
    INTEGER(KIND = idx_k) :: base, remainder
    base = global_size / mpi_comm_size
    remainder = MOD(global_size, mpi_comm_size)
    IF (mpi_rank < remainder) THEN
      local_size = base + 1_idx_k
    ELSE
      local_size = base
    END IF
  END FUNCTION compute_local_size
  FUNCTION compute_local_offset(global_size, mpi_comm_size, mpi_rank) RESULT(offset)
    IMPLICIT NONE
    INTEGER(KIND = idx_k), INTENT(IN) :: global_size, mpi_comm_size, mpi_rank
    INTEGER(KIND = idx_k) :: offset
    INTEGER(KIND = idx_k) :: base, remainder
    base = global_size / mpi_comm_size
    remainder = MOD(global_size, mpi_comm_size)
    IF (mpi_rank < remainder) THEN
      offset = mpi_rank * (base + 1_idx_k)
    ELSE
      offset = remainder * (base + 1_idx_k) + (mpi_rank - remainder) * base
    END IF
  END FUNCTION compute_local_offset
  FUNCTION distributed_check_result(local_u, g_nn, node_start_idx, nnode_local, tol, rank) RESULT(global_failed)
    IMPLICIT NONE
    REAL(KIND = 8), DIMENSION(:), INTENT(IN) :: local_u
    INTEGER(KIND = idx_k), INTENT(IN) :: g_nn
    INTEGER(KIND = idx_k), INTENT(IN) :: node_start_idx
    INTEGER(KIND = idx_k), INTENT(IN) :: nnode_local
    REAL(KIND = 8), INTENT(IN) :: tol
    INTEGER, INTENT(IN) :: rank
    INTEGER :: global_failed
    INTEGER :: local_failed
    INTEGER(KIND = idx_k) :: local_idx, global_idx, i, j
    REAL(KIND = 8) :: expected_value, diff
    INTEGER :: mpi_ierr
    local_failed = 0
    DO local_idx = 1_idx_k, nnode_local
      global_idx = node_start_idx + local_idx - 1_idx_k
      j = global_idx / (g_nn - 1) + 1_idx_k
      i = MOD(global_idx, g_nn - 1) + 1_idx_k
      IF (((i == 1_idx_k) .AND. (j == 1_idx_k)) .OR. ((i == 1_idx_k) .AND. (j == (g_nn - 1))) .OR. ((i == (g_nn - 1)) .AND. (j == 1_idx_k)) .OR. ((i == (g_nn - 1)) .AND. (j == (g_nn - 1)))) THEN
        expected_value = 0.625_8
      ELSE IF (((i == 1_idx_k) .AND. (j == 2_idx_k)) .OR. ((i == 2_idx_k) .AND. (j == 1_idx_k)) .OR. ((i == 1_idx_k) .AND. (j == (g_nn - 2))) .OR. ((i == 2_idx_k) .AND. (j == (g_nn - 1))) .OR. ((i == (g_nn - 2)) .AND. (j == 1_idx_k)) .OR. ((i == (g_nn - 1)) .AND. (j == 2_idx_k)) .OR. ((i == (g_nn - 2)) .AND. (j == (g_nn - 1))) .OR. ((i == (g_nn - 1)) .AND. (j == (g_nn - 2)))) THEN
        expected_value = 0.4375_8
      ELSE IF (((i == 2_idx_k) .AND. (j == 2_idx_k)) .OR. ((i == 2_idx_k) .AND. (j == (g_nn - 2))) .OR. ((i == (g_nn - 2)) .AND. (j == 2_idx_k)) .OR. ((i == (g_nn - 2)) .AND. (j == (g_nn - 2)))) THEN
        expected_value = 0.125_8
      ELSE IF ((i == 1_idx_k) .OR. (j == 1_idx_k) .OR. (i == (g_nn - 1)) .OR. (j == (g_nn - 1))) THEN
        expected_value = 0.3750_8
      ELSE IF ((i == 2_idx_k) .OR. (j == 2_idx_k) .OR. (i == (g_nn - 2)) .OR. (j == (g_nn - 2))) THEN
        expected_value = 0.0625_8
      ELSE
        expected_value = 0.0_8
      END IF
      diff = ABS(local_u(local_idx) - expected_value)
      IF (diff > tol) THEN
        PRINT *, "Validation Failure on rank ", rank, ": i=", i, ", j=", j, ", expected=", expected_value, ", actual=", local_u(local_idx), ", diff=", diff
        local_failed = 1
        EXIT
      END IF
    END DO
    CALL MPI_Allreduce(local_failed, global_failed, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, mpi_ierr)
    IF (mpi_ierr /= MPI_SUCCESS) THEN
      PRINT *, "Rank ", rank, ": MPI_Allreduce error in validation!"
      global_failed = 1
    END IF
    IF (rank == 0) THEN
      IF (global_failed == 0) THEN
        PRINT *, "Distributed results check PASSED on all ranks!"
      ELSE
        PRINT *, "Distributed results check FAILED on at least one rank!"
      END IF
    END IF
  END FUNCTION distributed_check_result
END PROGRAM jac_distributed