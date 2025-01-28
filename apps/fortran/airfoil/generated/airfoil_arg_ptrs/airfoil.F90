PROGRAM airfoil
  USE op2_kernels
  USE op2_fortran_declarations
  USE op2_fortran_rt_support
  USE airfoil_constants
  USE airfoil_kernels
  USE, INTRINSIC :: iso_c_binding
  IMPLICIT NONE
  INTEGER(KIND = 4), PARAMETER :: file_id = 1
  CHARACTER(LEN = *), PARAMETER :: file_name = "new_grid.dat"
  CHARACTER(LEN = *), PARAMETER :: file_name_h5 = "new_grid.h5"
  INTEGER(KIND = 4), PARAMETER :: niter = 1000
  INTEGER(KIND = 4) :: iter, k
  INTEGER(KIND = 4) :: nnode, ncell, nbedge, nedge
  INTEGER(KIND = 4) :: ncell_total
  INTEGER(KIND = 4), DIMENSION(*) :: ecell, bound, edge, bedge, becell, cell
  REAL(KIND = 8), DIMENSION(*) :: x, q, qold, adt, res
  POINTER(ptr_ecell, ecell), (ptr_bound, bound), (ptr_edge, edge), (ptr_bedge, bedge), (ptr_becell, becell), (ptr_cell, cell)
  POINTER(ptr_x, x), (ptr_q, q), (ptr_qold, qold), (ptr_adt, adt), (ptr_res, res)
  REAL(KIND = 8), DIMENSION(2) :: rms
  REAL(KIND = 8) :: maxerr
  INTEGER(KIND = 4) :: errloc
  TYPE(op_set) :: nodes, edges, bedges, cells
  TYPE(op_map) :: pedge, pecell, pcell, pbedge, pbecell
  TYPE(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res
  REAL(KIND = 8) :: diff
  PRINT *, "Reading input file"
  CALL read_input
  PRINT *, ncell
  CALL op_init_base(0, 0)
  CALL op_timing2_start("Airfoil")
  CALL op_print("Declaring OP2 sets")
  CALL op_decl_set(nnode, nodes, "nodes")
  CALL op_decl_set(nedge, edges, "edges")
  CALL op_decl_set(nbedge, bedges, "bedges")
  CALL op_decl_set(ncell, cells, "cells")
  CALL op_print("Declaring OP2 maps")
  CALL op_decl_map(edges, nodes, 2, edge, pedge, "pedge")
  CALL op_decl_map(edges, cells, 2, ecell, pecell, "pecell")
  CALL op_decl_map(bedges, nodes, 2, bedge, pbedge, "pbedge")
  CALL op_decl_map(bedges, cells, 1, becell, pbecell, "pbecell")
  CALL op_decl_map(cells, nodes, 4, cell, pcell, "pcell")
  CALL op_print("Declaring OP2 data")
  CALL op_decl_dat(bedges, 1, "integer(4)", bound, p_bound, "p_bound")
  CALL op_decl_dat(nodes, 2, "real(8)", x, p_x, "p_x")
  CALL op_decl_dat(cells, 4, "real(8)", q, p_q, "p_q")
  CALL op_decl_dat(cells, 4, "real(8)", qold, p_qold, "p_qold")
  CALL op_decl_dat(cells, 1, "real(8)", adt, p_adt, "p_adt")
  CALL release_buffers
  CALL op_print("Declaring OP2 constants")
  CALL op_decl_const_gam(gam, 1)
  CALL op_decl_const_gm1(gm1, 1)
  CALL op_decl_const_cfl(cfl, 1)
  CALL op_decl_const_eps(eps, 1)
  CALL op_decl_const_mach(mach, 1)
  CALL op_decl_const_alpha(alpha, 1)
  CALL op_decl_const_qinf(qinf, 4)
  CALL op_partition("PARMETIS", "KWAY", edges, pecell, p_x)
  CALL op_timing2_enter("Main computation")
  CALL op_decl_dat_temp(cells, 4, "real(8)", p_res, "p_res")
  ncell_total = op_get_size(cells)
  DO iter = 1, niter
    CALL op2_k_airfoil_1_save_soln("save_soln", cells, op_arg_dat(p_q, - 1, OP_ID, 4, "real(8)", OP_READ), op_arg_dat(p_qold, - 1, OP_ID, 4, "real(8)", OP_WRITE))
    DO k = 1, 2
      CALL op2_k_airfoil_2_adt_calc("adt_calc", cells, op_arg_dat(p_x, 1, pcell, 2, "real(8)", OP_READ), op_arg_dat(p_x, 2, pcell, 2, "real(8)", OP_READ), op_arg_dat(p_x, 3, pcell, 2, "real(8)", OP_READ), op_arg_dat(p_x, 4, pcell, 2, "real(8)", OP_READ), op_arg_dat(p_q, - 1, OP_ID, 4, "real(8)", OP_READ), op_arg_dat(p_adt, - 1, OP_ID, 1, "real(8)", OP_WRITE))
      CALL op2_k_airfoil_3_res_calc("res_calc", edges, op_arg_dat(p_x, 1, pedge, 2, "real(8)", OP_READ), op_arg_dat(p_x, 2, pedge, 2, "real(8)", OP_READ), op_arg_dat(p_q, 1, pecell, 4, "real(8)", OP_READ), op_arg_dat(p_q, 2, pecell, 4, "real(8)", OP_READ), op_arg_dat(p_adt, 1, pecell, 1, "real(8)", OP_READ), op_arg_dat(p_adt, 2, pecell, 1, "real(8)", OP_READ), op_arg_dat(p_res, 1, pecell, 4, "real(8)", OP_INC), op_arg_dat(p_res, 2, pecell, 4, "real(8)", OP_INC))
      CALL op2_k_airfoil_4_bres_calc("bres_calc", bedges, op_arg_dat(p_x, 1, pbedge, 2, "real(8)", OP_READ), op_arg_dat(p_x, 2, pbedge, 2, "real(8)", OP_READ), op_arg_dat(p_q, 1, pbecell, 4, "real(8)", OP_READ), op_arg_dat(p_adt, 1, pbecell, 1, "real(8)", OP_READ), op_arg_dat(p_res, 1, pbecell, 4, "real(8)", OP_INC), op_arg_dat(p_bound, - 1, OP_ID, 1, "integer(4)", OP_READ))
      rms = 0.0
      maxerr = 0.0
      errloc = 0
      CALL op2_k_airfoil_5_update("update", cells, op_arg_dat(p_qold, - 1, OP_ID, 4, "real(8)", OP_READ), op_arg_dat(p_q, - 1, OP_ID, 4, "real(8)", OP_WRITE), op_arg_dat(p_res, - 1, OP_ID, 4, "real(8)", OP_RW), op_arg_dat(p_adt, - 1, OP_ID, 1, "real(8)", OP_READ), op_arg_gbl(rms, 2, "real(8)", OP_INC), op_arg_gbl(maxerr, 1, "real(8)", OP_MAX), op_arg_idx(- 1, OP_ID), op_arg_info(errloc, 1, "integer(4)", 6))
    END DO
    rms(2) = SQRT(rms(2) / REAL(ncell_total))
    IF (op_is_root() == 1 .AND. MOD(iter, 100) == 0) THEN
      WRITE(*, "(4X, I0, E16.7, 4X, A, E16.7, A, I0)") iter, rms(2), "max err: ", maxerr, " at ", errloc
    END IF
  END DO
  iter = op_free_dat_temp(p_res)
  CALL op_timing2_finish
  IF (op_is_root() == 1) PRINT *
  CALL op_timing2_output
  IF (op_is_root() == 1 .AND. niter == 1000 .AND. ncell_total == 720000) THEN
    diff = ABS((100.0_8 * (rms(2) / 0.0001060114637578_8)) - 100.0_8)
    WRITE(*, "(A, I0, A, E16.7, A)") " Test problem with ", ncell_total, " cells is within ", diff, "% of the expected solution"
    IF (diff < 0.00001_8) THEN
      PRINT *, "Test PASSED"
    ELSE
      PRINT *, "Test FAILED"
    END IF
  END IF
  CALL op_exit
  CONTAINS
  SUBROUTINE read_input
    IMPLICIT NONE
    INTEGER(KIND = 4) :: i
    OPEN(UNIT = file_id, FILE = file_name)
    READ(file_id, *) nnode, ncell, nedge, nbedge
    CALL op_memalloc(ptr_x, 2 * nnode * INT(sizeof(x(1))))
    CALL op_memalloc(ptr_cell, 4 * ncell * INT(sizeof(cell(1))))
    CALL op_memalloc(ptr_edge, 2 * nedge * INT(sizeof(edge(1))))
    CALL op_memalloc(ptr_ecell, 2 * nedge * INT(sizeof(ecell(1))))
    CALL op_memalloc(ptr_bedge, 2 * nbedge * INT(sizeof(bedge(1))))
    CALL op_memalloc(ptr_becell, nbedge * INT(sizeof(becell(1))))
    CALL op_memalloc(ptr_bound, nbedge * INT(sizeof(bound(1))))
    DO i = 1, nnode
      READ(file_id, *) x(2 * (i - 1) + 1), x(2 * (i - 1) + 2)
    END DO
    DO i = 1, ncell
      READ(file_id, *) cell(4 * (i - 1) + 1), cell(4 * (i - 1) + 2), cell(4 * (i - 1) + 3), cell(4 * (i - 1) + 4)
    END DO
    DO i = 1, nedge
      READ(file_id, *) edge(2 * (i - 1) + 1), edge(2 * (i - 1) + 2), ecell(2 * (i - 1) + 1), ecell(2 * (i - 1) + 2)
    END DO
    DO i = 1, nbedge
      READ(file_id, *) bedge(2 * (i - 1) + 1), bedge(2 * (i - 1) + 2), becell(i), bound(i)
    END DO
    CLOSE(UNIT = file_id)
    DO i = 1, ncell
      q(4 * (i - 1) + 1 : 4 * (i - 1) + 4) = qinf
    END DO
    qold(: 4 * ncell) = 0.0_8
    res(: 4 * ncell) = 0.0_8
    adt(: ncell) = 0.0_8
  END SUBROUTINE
  SUBROUTINE release_buffers
  END SUBROUTINE
END PROGRAM