program AIRFOIL
  use OP2_FORTRAN_HDF5_DECLARATIONS
  use OP2_Fortran_Reference
  use OP2_CONSTANTS
  use AIRFOIL_SEQ
  use IO
  use, intrinsic :: ISO_C_BINDING

  implicit none

  intrinsic :: sqrt, real

  integer(4) :: iter, k, i

  integer(4), parameter :: maxnode = 9900
  integer(4), parameter :: maxcell = (9702+1)
  integer(4), parameter :: maxedge = 19502

  integer(4), parameter :: iterationNumber = 1000

  integer(4) :: nnode, ncell, nbedge, nedge, niter, qdim
  real(8) :: ncellr

  ! profiling
  real(kind=c_double) :: startTime = 0
  real(kind=c_double) :: endTime = 0

  ! integer references (valid inside the OP2 library) for op_set
  type(op_set) :: nodes, edges, bedges, cells

  ! integer references (valid inside the OP2 library) for pointers between data sets
  type(op_map) :: pedge, pecell, pcell, pbedge, pbecell
  type(op_map) :: m_test

  ! integer reference (valid inside the OP2 library) for op_data
  type(op_dat) :: p_bound, p_x, p_q, p_qold, p_adt, p_res
  type(op_dat) :: p_test


  ! arrays used in data
  integer(4), dimension(:), allocatable, target :: ecell, bound, edge, bedge, becell, cell
  real(8), dimension(1:2) :: rms

  integer(4) :: debugiter, retDebug
  real(8) :: datad

  ! local variables for constant initialization
  real(8) :: p, r, u, e

  integer(4) :: status

  ! for validation
  REAL(KIND=8) :: diff
  integer(4):: ncelli

  ! OP initialisation
  call op_init_base (0,0)

  ! declare sets, pointers, datasets and global constants (for now, no new partition info)
  call op_print ("Declaring OP2 sets")
  call op_decl_set_hdf5 ( nnode, nodes, 'new_grid.h5', 'nodes' )
  call op_decl_set_hdf5 ( nedge, edges, 'new_grid.h5', 'edges' )
  call op_decl_set_hdf5 ( nbedge, bedges, 'new_grid.h5', 'bedges' )
  call op_decl_set_hdf5 ( ncell, cells, 'new_grid.h5', 'cells' )

  call op_print ("Declaring OP2 maps")
  call op_decl_map_hdf5 ( edges, nodes, 2, pedge, 'new_grid.h5', 'pedge', status )
  call op_decl_map_hdf5 ( edges, cells, 2, pecell, 'new_grid.h5', 'pecell', status )
  call op_decl_map_hdf5 ( bedges, nodes, 2, pbedge, 'new_grid.h5', 'pbedge', status )
  call op_decl_map_hdf5 ( bedges, cells, 1, pbecell, 'new_grid.h5', 'pbecell', status )
  call op_decl_map_hdf5 ( cells, nodes, 4, pcell, 'new_grid.h5', 'pcell', status )

  write (*,*) 'size of pcell ', status
  if (status .lt. 0)then
    write (*,*) 'pcell does do not exist', status
  end if

  call op_decl_map_hdf5 ( cells, nodes, 4, m_test, 'new_grid.h5', 'm_test', status )

  if (status .lt. 0)then
    write (*,*) 'm_test does do not exist', status
  end if

  call op_print ("Declaring OP2 data")
  call op_decl_dat_hdf5 ( bedges, 1, p_bound, 'integer', 'new_grid.h5', 'p_bound', status )
  call op_decl_dat_hdf5 ( nodes, 2, p_x, 'real(8)', 'new_grid.h5', 'p_x' , status)
  call op_decl_dat_hdf5 ( cells, 4, p_q, 'real(8)', 'new_grid.h5', 'p_q' , status)
  call op_decl_dat_hdf5 ( cells, 4, p_qold, 'real(8)', 'new_grid.h5', 'p_qold' , status)
  call op_decl_dat_hdf5 ( cells, 1, p_adt, 'real(8)', 'new_grid.h5', 'p_adt' , status)
  call op_decl_dat_hdf5 ( cells, 4, p_res, 'real(8)', 'new_grid.h5', 'p_res' , status)

  write (*,*) 'size of p_res ', status
  if (status .lt. 0)then
    write (*,*) 'p_res does do not exist', status
  end if

  call op_decl_dat_hdf5 ( cells, 4, p_test, 'real(8)', 'new_grid.h5', 'p_test' , status)

  if (status .lt. 0)then
    write (*,*) 'p_test does do not exist', status
  end if

  call op_print ("Declaring OP2 constants")
  call op_decl_const(gam, 1, 'gam')
  call op_decl_const(gm1, 1, 'gm1')
  call op_decl_const(cfl, 1, 'cfl')
  call op_decl_const(eps, 1, 'eps')
  call op_decl_const(mach, 1, 'mach')
  call op_decl_const(alpha, 1, 'alpha')
  call op_decl_const(qinf, 4, 'qinf')

  call op_print ('Initialising constants')
  call initialise_constants ( )
  !call op_dump_to_hdf5("new_grid_out.h5");

  call op_partition ('PTSCOTCH','KWAY', edges, pecell, p_x)

  ncelli  = op_get_size(cells)
  ncellr = real(ncelli)

  ! start timer
  call op_timers ( startTime )

  ! main time-marching loop

  do niter = 1, iterationNumber

     call op_par_loop_2 ( save_soln, cells, &
                       & op_arg_dat (p_q,    -1, OP_ID, 4,"real(8)", OP_READ), &
                       & op_arg_dat (p_qold, -1, OP_ID, 4,"real(8)", OP_WRITE))

    ! predictor/corrector update loop

    do k = 1, 2

      ! calculate area/timstep
      call op_par_loop_6 ( adt_calc, cells, &
                         & op_arg_dat (p_x,    1, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_x,    2, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_x,    3, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_x,    4, pcell, 2,"real(8)", OP_READ), &
                         & op_arg_dat (p_q,   -1, OP_ID, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt, -1, OP_ID, 1,"real(8)", OP_WRITE))

      ! calculate flux residual
      call op_par_loop_8 ( res_calc, edges, &
                         & op_arg_dat (p_x,    1, pedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_x,    2, pedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_q,    1, pecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_q,    2, pecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt,  1, pecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt,  2, pecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (p_res,  1, pecell, 4,"real(8)", OP_INC),  &
                         & op_arg_dat (p_res,  2, pecell, 4,"real(8)", OP_INC))

      call op_par_loop_6 ( bres_calc, bedges, &
                         & op_arg_dat (p_x,      1, pbedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_x,      2, pbedge, 2,"real(8)",  OP_READ), &
                         & op_arg_dat (p_q,      1, pbecell, 4,"real(8)", OP_READ), &
                         & op_arg_dat (p_adt,    1, pbecell, 1,"real(8)", OP_READ), &
                         & op_arg_dat (p_res,    1, pbecell, 4,"real(8)", OP_INC),  &
                         & op_arg_dat (p_bound, -1, OP_ID, 1,"integer", OP_READ))

      ! update flow field

      rms(1:2) = 0.0

      call op_par_loop_5 ( update, cells, &
                         & op_arg_dat (p_qold, -1, OP_ID, 4,"real(8)",  OP_READ),  &
                         & op_arg_dat (p_q,    -1, OP_ID, 4,"real(8)",  OP_WRITE), &
                         & op_arg_dat (p_res,  -1, OP_ID, 4,"real(8)",  OP_RW),    &
                         & op_arg_dat (p_adt,  -1, OP_ID, 1,"real(8)",  OP_READ),  &
                         & op_arg_gbl (rms, 2, "real(8)", OP_INC))


    end do ! internal loop

    rms(2) = sqrt ( rms(2) / ncellr )

    if (op_is_root() .eq. 1) then
      if (mod(niter,100) .eq. 0) then
        write (*,*) niter,"  ",rms(2)
      end if
      if ((mod(niter,1000) .eq. 0) .AND. (ncelli == 720000) ) then
        diff=ABS((100.0_8*(rms(2)/0.0001060114637578_8))-100.0_8)
        !write (*,*) niter,"  ",rms(2)
        WRITE(*,'(a,i0,a,e16.7,a)')"Test problem with ", ncelli , &
        & " cells is within ",diff,"% of the expected solution"
        if(diff.LT.0.00001) THEN
          WRITE(*,*)"This test is considered PASSED"
        else
          WRITE(*,*)"This test is considered FAILED"
        endif
      end if
    end if



  end do ! external loop

  call op_timers ( endTime )
  call op_timing_output ()
  if (op_is_root() .eq. 1) then
    write (*,*) 'Max total runtime =', endTime - startTime,'seconds'
  end if
  call op_exit (  )
end program AIRFOIL
