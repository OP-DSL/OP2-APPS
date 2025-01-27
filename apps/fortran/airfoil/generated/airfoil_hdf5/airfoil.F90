PROGRAM airfoil
  USE op2_kernels
  USE op2_fortran_hdf5_declarations
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
  INTEGER(KIND = 4) :: stat
  REAL(KIND = 8), DIMENSION(2) :: rms
  REAL(KIND = 8) :: maxerr
  INTEGER(KIND = 4) :: errloc
  TYPE(op_set) :: nodes, edges, bedges, cells
  TYPE(op_map) :: pedge, pecell, pcell, pbedge, pbecell
  TYPE(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res
  REAL(KIND = 8) :: diff
  CALL op_init_base(0, 0)
  CALL op_timing2_start("Airfoil")
  CALL op_print("Declaring OP2 sets (HDF5)")
  CALL op_decl_set_hdf5(nnode, nodes, file_name_h5, "nodes")
  CALL op_decl_set_hdf5(nedge, edges, file_name_h5, "edges")
  CALL op_decl_set_hdf5(nbedge, bedges, file_name_h5, "bedges")
  CALL op_decl_set_hdf5(ncell, cells, file_name_h5, "cells")
  CALL op_print("Declaring OP2 maps (HDF5)")
  CALL op_decl_map_hdf5(edges, nodes, 2, pedge, file_name_h5, "pedge", stat)
  CALL op_decl_map_hdf5(edges, cells, 2, pecell, file_name_h5, "pecell", stat)
  CALL op_decl_map_hdf5(bedges, nodes, 2, pbedge, file_name_h5, "pbedge", stat)
  CALL op_decl_map_hdf5(bedges, cells, 1, pbecell, file_name_h5, "pbecell", stat)
  CALL op_decl_map_hdf5(cells, nodes, 4, pcell, file_name_h5, "pcell", stat)
  CALL op_print("Declaring OP2 data (HDF5)")
  CALL op_decl_dat_hdf5(bedges, 1, p_bound, "integer(4)", file_name_h5, "p_bound", stat)
  CALL op_decl_dat_hdf5(nodes, 2, p_x, "real(8)", file_name_h5, "p_x", stat)
  CALL op_decl_dat_hdf5(cells, 4, p_q, "real(8)", file_name_h5, "p_q", stat)
  CALL op_decl_dat_hdf5(cells, 4, p_qold, "real(8)", file_name_h5, "p_qold", stat)
  CALL op_decl_dat_hdf5(cells, 1, p_adt, "real(8)", file_name_h5, "p_adt", stat)
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
END PROGRAM