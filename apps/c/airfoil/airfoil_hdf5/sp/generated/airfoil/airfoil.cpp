/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
//     Nonlinear airfoil lift calculation
//
//     Written by Mike Giles, 2010-2011, based on FORTRAN code
//     by Devendra Ghate and Mike Giles, 2005
//

//
// standard headers
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// global constants

float gam, gm1, cfl, eps, mach, alpha, qinf[4];

//
// OP header file
//

#include "op_lib_cpp.h"

#ifdef OPENACC
#ifdef __cplusplus
extern "C" {
#endif
#endif

void op_par_loop_airfoil_1_save_soln(char const *, op_set, op_arg, op_arg);

void op_par_loop_airfoil_2_adt_calc(char const *, op_set, op_arg, op_arg, op_arg, op_arg, op_arg, op_arg);

void op_par_loop_airfoil_3_res_calc(char const *, op_set, op_arg, op_arg, op_arg, op_arg, op_arg, op_arg, op_arg, op_arg);

void op_par_loop_airfoil_4_bres_calc(char const *, op_set, op_arg, op_arg, op_arg, op_arg, op_arg, op_arg);

void op_par_loop_airfoil_5_update(char const *, op_set, op_arg, op_arg, op_arg, op_arg, op_arg);

#ifdef OPENACC
#ifdef __cplusplus
}
#endif
#endif

//
// kernel routines for parallel loops
//

#include "adt_calc.h"
#include "bres_calc.h"
#include "res_calc.h"
#include "save_soln.h"
#include "update.h"

// main program

int main(int argc, char **argv) {
  // OP initialisation
  op_init(argc, argv, 2);

  int niter;
  float rms;

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  // set constants and initialise flow field and residual
  op_printf("initialising flow field \n");

  char file[] = "new_grid.h5";

  // declare sets, pointers, datasets and global constants

  op_set nodes = op_decl_set_hdf5(file, "nodes");
  op_set edges = op_decl_set_hdf5(file, "edges");
  op_set bedges = op_decl_set_hdf5(file, "bedges");
  op_set cells = op_decl_set_hdf5(file, "cells");

  op_map pedge = op_decl_map_hdf5(edges, nodes, 2, file, "pedge");
  op_map pecell = op_decl_map_hdf5(edges, cells, 2, file, "pecell");
  op_map pbedge = op_decl_map_hdf5(bedges, nodes, 2, file, "pbedge");
  op_map pbecell = op_decl_map_hdf5(bedges, cells, 1, file, "pbecell");
  op_map pcell = op_decl_map_hdf5(cells, nodes, 4, file, "pcell");

  op_dat p_bound = op_decl_dat_hdf5(bedges, 1, "int", file, "p_bound");
  op_dat p_x = op_decl_dat_hdf5(nodes, 2, "float", file, "p_x");
  op_dat p_q = op_decl_dat_hdf5(cells, 4, "float", file, "p_q");
  op_dat p_qold = op_decl_dat_hdf5(cells, 4, "float", file, "p_qold");
  op_dat p_adt = op_decl_dat_hdf5(cells, 1, "float", file, "p_adt");
  op_dat p_res = op_decl_dat_hdf5(cells, 4, "float", file, "p_res");

  op_get_const_hdf5("gam", 1, "float", (char *)&gam, "new_grid.h5");
  op_get_const_hdf5("gm1", 1, "float", (char *)&gm1, "new_grid.h5");
  op_get_const_hdf5("cfl", 1, "float", (char *)&cfl, "new_grid.h5");
  op_get_const_hdf5("eps", 1, "float", (char *)&eps, "new_grid.h5");
  op_get_const_hdf5("mach", 1, "float", (char *)&mach, "new_grid.h5");
  op_get_const_hdf5("alpha", 1, "float", (char *)&alpha, "new_grid.h5");
  op_get_const_hdf5("qinf", 4, "float", (char *)&qinf, "new_grid.h5");

  op_decl_const2("gam", 1, "float", &gam);
  op_decl_const2("gm1", 1, "float", &gm1);
  op_decl_const2("cfl", 1, "float", &cfl);
  op_decl_const2("eps", 1, "float", &eps);
  op_decl_const2("mach", 1, "float", &mach);
  op_decl_const2("alpha", 1, "float", &alpha);
  op_decl_const2("qinf", 4, "float", qinf);

  if (op_is_root())
    op_diagnostic_output();

  // trigger partitioning and halo creation routines
  op_partition("PTSCOTCH", "KWAY", edges, pecell, p_x);
  // op_partition("PARMETIS", "KWAY", edges, pecell, p_x);

  int g_ncell = op_get_size(cells);

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // main time-marching loop

  niter = 1000;

  for (int iter = 1; iter <= niter; iter++) {

    //  save old flow solution

    op_par_loop_airfoil_1_save_soln("save_soln", cells,
                op_arg_dat(p_q, -1, OP_ID, 4, "float", OP_READ),
                op_arg_dat(p_qold, -1, OP_ID, 4, "float", OP_WRITE));

    //  predictor/corrector update loop

    for (int k = 0; k < 2; k++) {

      //    calculate area/timstep

      op_par_loop_airfoil_2_adt_calc("adt_calc", cells,
                  op_arg_dat(p_x, 0, pcell, 2, "float", OP_READ),
                  op_arg_dat(p_x, 1, pcell, 2, "float", OP_READ),
                  op_arg_dat(p_x, 2, pcell, 2, "float", OP_READ),
                  op_arg_dat(p_x, 3, pcell, 2, "float", OP_READ),
                  op_arg_dat(p_q, -1, OP_ID, 4, "float", OP_READ),
                  op_arg_dat(p_adt, -1, OP_ID, 1, "float", OP_WRITE));

      //    calculate flux residual

      op_par_loop_airfoil_3_res_calc("res_calc", edges,
                  op_arg_dat(p_x, 0, pedge, 2, "float", OP_READ),
                  op_arg_dat(p_x, 1, pedge, 2, "float", OP_READ),
                  op_arg_dat(p_q, 0, pecell, 4, "float", OP_READ),
                  op_arg_dat(p_q, 1, pecell, 4, "float", OP_READ),
                  op_arg_dat(p_adt, 0, pecell, 1, "float", OP_READ),
                  op_arg_dat(p_adt, 1, pecell, 1, "float", OP_READ),
                  op_arg_dat(p_res, 0, pecell, 4, "float", OP_INC),
                  op_arg_dat(p_res, 1, pecell, 4, "float", OP_INC));

      op_par_loop_airfoil_4_bres_calc("bres_calc", bedges,
                  op_arg_dat(p_x, 0, pbedge, 2, "float", OP_READ),
                  op_arg_dat(p_x, 1, pbedge, 2, "float", OP_READ),
                  op_arg_dat(p_q, 0, pbecell, 4, "float", OP_READ),
                  op_arg_dat(p_adt, 0, pbecell, 1, "float", OP_READ),
                  op_arg_dat(p_res, 0, pbecell, 4, "float", OP_INC),
                  op_arg_dat(p_bound, -1, OP_ID, 1, "int", OP_READ));

      //    update flow field

      rms = 0.0;

      op_par_loop_airfoil_5_update("update", cells,
                  op_arg_dat(p_qold, -1, OP_ID, 4, "float", OP_READ),
                  op_arg_dat(p_q, -1, OP_ID, 4, "float", OP_WRITE),
                  op_arg_dat(p_res, -1, OP_ID, 4, "float", OP_RW),
                  op_arg_dat(p_adt, -1, OP_ID, 1, "float", OP_READ),
                  op_arg_gbl(&rms, 1, "float", OP_INC));
    }

    //  print iteration history

    rms = sqrtf(rms / (float)g_ncell);

    if (iter % 100 == 0)
      op_printf(" %d  %10.5e \n", iter, rms);
    if (iter % 1000 == 0 &&
        g_ncell == 720000) { // defailt mesh -- for validation testing
      op_printf(" %d  %3.16f \n", iter, rms);
      float diff = fabsf((100.0 * (rms / 0.000105987)) - 100.0);
      op_printf("\n\nTest problem with %d cells is within %3.15E %% of the "
                "expected solution\n",
                720000, diff);
      if (diff < 0.1) {
        op_printf("This test is considered PASSED\n");
      } else {
        op_printf("This test is considered FAILED\n");
      }
    }
  }

  op_timers(&cpu_t2, &wall_t2);

  op_timing_output();
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);
  op_exit();
}