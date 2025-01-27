PROGRAM reduction
  USE op2_kernels
  USE op2_fortran_declarations
  USE op2_fortran_rt_support
  USE, INTRINSIC :: ISO_C_BINDING
  IMPLICIT NONE
  INTEGER(KIND = 4), PARAMETER :: file_id = 1
  CHARACTER(LEN = *), PARAMETER :: file_name = "new_grid.dat"
  CHARACTER(LEN = *), PARAMETER :: file_name_h5 = "new_grid.h5"
  INTEGER(KIND = 4) :: nnode, ncell, nedge
  INTEGER(KIND = 4) :: ncell_total, nedge_total
  INTEGER(KIND = 4), DIMENSION(:), ALLOCATABLE, TARGET :: ecell
  REAL(KIND = 8), DIMENSION(:), ALLOCATABLE, TARGET :: res
  TYPE(op_set) :: edges, cells
  TYPE(op_map) :: pecell
  TYPE(op_dat) :: p_res, p_dummy
  REAL(KIND = c_double) :: start_time, end_time
  INTEGER(KIND = 4) :: i, cell_count_result, edge_count_result
  INTEGER(KIND = 4) :: dummy_int
  REAL(KIND = 8) :: dummy_real
  CALL op_init_base(0, 0)
  OPEN(UNIT = file_id, FILE = file_name)
  READ(file_id, *) nnode, ncell, nedge, dummy_int
  ALLOCATE(ecell(2 * nedge))
  DO i = 1, nnode
    READ(file_id, *) dummy_real, dummy_real
  END DO
  DO i = 1, ncell
    READ(file_id, *) dummy_int, dummy_int, dummy_int, dummy_int
  END DO
  DO i = 1, nedge
    READ(file_id, *) dummy_int, dummy_int, ecell(2 * (i - 1) + 1), ecell(2 * (i - 1) + 2)
  END DO
  CLOSE(UNIT = file_id)
  ALLOCATE(res(4 * ncell))
  res = 0.0
  CALL op_decl_set(nedge, edges, "edges")
  CALL op_decl_set(ncell, cells, "cells")
  CALL op_decl_map(edges, cells, 2, ecell, pecell, "pecell")
  CALL op_decl_dat(cells, 4, "real(8)", res, p_res, "p_res")
  DEALLOCATE(ecell)
  DEALLOCATE(res)
  CALL op_partition("PTSCOTCH", "KWAY", edges, pecell, p_dummy)
  CALL op_timers(start_time)
  ncell_total = op_get_size(cells)
  nedge_total = op_get_size(edges)
  cell_count_result = 0
  edge_count_result = 0
  CALL op2_k_reduction_1_cell_count("cell_count", cells, op_arg_dat(p_res, - 1, OP_ID, 4, "real(8)", OP_RW), op_arg_gbl(cell_count_result, 1, "integer(4)", OP_INC))
  CALL op2_k_reduction_2_edge_count("edge_count", edges, op_arg_dat(p_res, 1, pecell, 4, "real(8)", OP_RW), op_arg_gbl(edge_count_result, 1, "integer(4)", OP_INC))
  CALL op_timers(end_time)
  CALL op_timing_output
  IF (op_is_root() == 1) THEN
    PRINT *
    PRINT *, "Direct reduction: cell count = ", cell_count_result, ", target = ", ncell_total
    PRINT *, "Indirect reduction: edge count = ", edge_count_result, ", target = ", nedge_total
    PRINT *
    IF (cell_count_result == ncell_total .AND. edge_count_result == nedge_total) THEN
      PRINT *, "Test PASSED"
    ELSE
      PRINT *, "Test FAILED"
    END IF
    PRINT *
    PRINT *, 'Time = ', end_time - start_time, 'seconds'
  END IF
  CALL op_exit
  CONTAINS
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
END PROGRAM