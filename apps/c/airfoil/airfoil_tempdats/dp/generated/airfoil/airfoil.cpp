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

double gam, gm1, cfl, eps, mach, alpha, qinf[4];

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

  int *becell, *ecell, *bound, *bedge, *edge, *cell;
  double *x, *q, *qold, *adt, *res;

  int nnode, ncell, nedge, nbedge, niter;
  double rms;

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  // read in grid

  op_printf("reading in grid \n");

  FILE *fp;
  if ((fp = fopen("./new_grid.dat", "r")) == NULL) {
    op_printf("can't open file new_grid.dat\n");
    exit(-1);
  }

  if (fscanf(fp, "%d %d %d %d \n", &nnode, &ncell, &nedge, &nbedge) != 4) {
    op_printf("error reading from new_grid.dat\n");
    exit(-1);
  }

  cell = (int *)malloc(4 * ncell * sizeof(int));
  edge = (int *)malloc(2 * nedge * sizeof(int));
  ecell = (int *)malloc(2 * nedge * sizeof(int));
  bedge = (int *)malloc(2 * nbedge * sizeof(int));
  becell = (int *)malloc(nbedge * sizeof(int));
  bound = (int *)malloc(nbedge * sizeof(int));

  x = (double *)malloc(2 * nnode * sizeof(double));
  q = (double *)malloc(4 * ncell * sizeof(double));
  qold = (double *)malloc(4 * ncell * sizeof(double));
  res = (double *)malloc(4 * ncell * sizeof(double));
  adt = (double *)malloc(ncell * sizeof(double));

  for (int n = 0; n < nnode; n++) {
    if (fscanf(fp, "%lf %lf \n", &x[2 * n], &x[2 * n + 1]) != 2) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < ncell; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &cell[4 * n], &cell[4 * n + 1],
               &cell[4 * n + 2], &cell[4 * n + 3]) != 4) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < nedge; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &edge[2 * n], &edge[2 * n + 1],
               &ecell[2 * n], &ecell[2 * n + 1]) != 4) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  for (int n = 0; n < nbedge; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &bedge[2 * n], &bedge[2 * n + 1],
               &becell[n], &bound[n]) != 4) {
      op_printf("error reading from new_grid.dat\n");
      exit(-1);
    }
  }

  fclose(fp);

  // set constants and initialise flow field and residual

  op_printf("initialising flow field \n");

  gam = 1.4f;
  gm1 = gam - 1.0f;
  cfl = 0.9f;
  eps = 0.05f;

  double mach = 0.4f;
  double alpha = 3.0f * atan(1.0f) / 45.0f;
  double p = 1.0f;
  double r = 1.0f;
  double u = sqrt(gam * p / r) * mach;
  double e = p / (r * gm1) + 0.5f * u * u;

  qinf[0] = r;
  qinf[1] = r * u;
  qinf[2] = 0.0f;
  qinf[3] = r * e;

  for (int n = 0; n < ncell; n++) {
    for (int m = 0; m < 4; m++) {
      q[4 * n + m] = qinf[m];
      res[4 * n + m] = 0.0f;
    }
  }

  // declare sets, pointers, datasets and global constants

  op_set nodes = op_decl_set(nnode, "nodes");
  op_set edges = op_decl_set(nedge, "edges");
  op_set bedges = op_decl_set(nbedge, "bedges");
  op_set cells = op_decl_set(ncell, "cells");

  op_map pedge = op_decl_map(edges, nodes, 2, edge, "pedge");
  free(edge);
  op_map pecell = op_decl_map(edges, cells, 2, ecell, "pecell");
  free(ecell);
  op_map pbedge = op_decl_map(bedges, nodes, 2, bedge, "pbedge");
  free(bedge);
  op_map pbecell = op_decl_map(bedges, cells, 1, becell, "pbecell");
  free(becell);
  op_map pcell = op_decl_map(cells, nodes, 4, cell, "pcell");
  free(cell);

  op_dat p_bound = op_decl_dat(bedges, 1, "int", bound, "p_bound");
  free(bound);
  op_dat p_x = op_decl_dat(nodes, 2, "double", x, "p_x");
  free(x);
  op_dat p_q = op_decl_dat(cells, 4, "double", q, "p_q");
  free(q);
  // op_dat p_qold  = op_decl_dat(cells ,4,"double",qold ,"p_qold");
  // op_dat p_adt   = op_decl_dat(cells ,1,"double",adt  ,"p_adt");
  // op_dat p_res   = op_decl_dat(cells ,4,"double",res  ,"p_res");

  // p_res, p_adt and p_qold  now declared as a temp op_dats during
  // the execution of the time-marching loop

  op_decl_const2("gam", 1, "double", &gam);
  op_decl_const2("gm1", 1, "double", &gm1);
  op_decl_const2("cfl", 1, "double", &cfl);
  op_decl_const2("eps", 1, "double", &eps);
  op_decl_const2("mach", 1, "double", &mach);
  op_decl_const2("alpha", 1, "double", &alpha);
  op_decl_const2("qinf", 4, "double", qinf);

  op_diagnostic_output();

  double g_ncell = op_get_size(cells);

  // initialise timers for total execution wall time
  op_timers(&cpu_t1, &wall_t1);

  // main time-marching loop

  niter = 1000;

  for (int iter = 1; iter <= niter; iter++) {

    double *tmp_elem = NULL;
    op_dat p_res = op_decl_dat_temp(cells, 4, "double", tmp_elem, "p_res");
    op_dat p_adt = op_decl_dat_temp(cells, 1, "double", tmp_elem, "p_adt");
    op_dat p_qold = op_decl_dat_temp(cells, 4, "double", qold, "p_qold");

    // save old flow solution

    op_par_loop_airfoil_1_save_soln("save_soln", cells,
                op_arg_dat(p_q, -1, OP_ID, 4, "double", OP_READ),
                op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_WRITE));

    // predictor/corrector update loop

    for (int k = 0; k < 2; k++) {

      // calculate area/timstep

      op_par_loop_airfoil_2_adt_calc("adt_calc", cells,
                  op_arg_dat(p_x, 0, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_x, 1, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_x, 2, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_x, 3, pcell, 2, "double", OP_READ),
                  op_arg_dat(p_q, -1, OP_ID, 4, "double", OP_READ),
                  op_arg_dat(p_adt, -1, OP_ID, 1, "double", OP_WRITE));

      // calculate flux residual

      op_par_loop_airfoil_3_res_calc("res_calc", edges,
                  op_arg_dat(p_x, 0, pedge, 2, "double", OP_READ),
                  op_arg_dat(p_x, 1, pedge, 2, "double", OP_READ),
                  op_arg_dat(p_q, 0, pecell, 4, "double", OP_READ),
                  op_arg_dat(p_q, 1, pecell, 4, "double", OP_READ),
                  op_arg_dat(p_adt, 0, pecell, 1, "double", OP_READ),
                  op_arg_dat(p_adt, 1, pecell, 1, "double", OP_READ),
                  op_arg_dat(p_res, 0, pecell, 4, "double", OP_INC),
                  op_arg_dat(p_res, 1, pecell, 4, "double", OP_INC));

      op_par_loop_airfoil_4_bres_calc("bres_calc", bedges,
                  op_arg_dat(p_x, 0, pbedge, 2, "double", OP_READ),
                  op_arg_dat(p_x, 1, pbedge, 2, "double", OP_READ),
                  op_arg_dat(p_q, 0, pbecell, 4, "double", OP_READ),
                  op_arg_dat(p_adt, 0, pbecell, 1, "double", OP_READ),
                  op_arg_dat(p_res, 0, pbecell, 4, "double", OP_INC),
                  op_arg_dat(p_bound, -1, OP_ID, 1, "int", OP_READ));

      // update flow field

      rms = 0.0;

      op_par_loop_airfoil_5_update("update", cells,
                  op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_READ),
                  op_arg_dat(p_q, -1, OP_ID, 4, "double", OP_WRITE),
                  op_arg_dat(p_res, -1, OP_ID, 4, "double", OP_RW),
                  op_arg_dat(p_adt, -1, OP_ID, 1, "double", OP_READ),
                  op_arg_gbl(&rms, 1, "double", OP_INC));
    }

    // print iteration history
    rms = sqrt(rms / (double)g_ncell);
    if (iter % 100 == 0)
      op_printf(" %d  %10.5e \n", iter, rms);

    if (iter % 1000 == 0 &&
        g_ncell == 720000) { // defailt mesh -- for validation testing
      // op_printf(" %d  %3.16f \n",iter,rms);
      double diff = fabs((100.0 * (rms / 0.0001060114637578)) - 100.0);
      op_printf("\n\nTest problem with %d cells is within %3.15E %% of the "
                "expected solution\n",
                720000, diff);
      if (diff < 0.00001) {
        op_printf("This test is considered PASSED\n");
      } else {
        op_printf("This test is considered FAILED\n");
      }
    }

    if (op_free_dat_temp(p_res) < 0)
      op_printf("Error: temporary op_dat %s cannot be removed\n", p_res->name);
    if (op_free_dat_temp(p_adt) < 0)
      op_printf("Error: temporary op_dat %s cannot be removed\n", p_adt->name);
    if (op_free_dat_temp(p_qold) < 0)
      op_printf("Error: temporary op_dat %s cannot be removed\n", p_qold->name);
  }

  op_timers(&cpu_t2, &wall_t2);
  op_timing_output();
  op_printf("Max total runtime = %f\n", wall_t2 - wall_t1);

  op_exit();
}