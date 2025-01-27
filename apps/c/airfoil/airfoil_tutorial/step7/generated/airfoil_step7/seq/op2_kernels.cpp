extern double gam;
extern double gm1;
extern double cfl;
extern double eps;
extern double mach;
extern double alpha;
extern double qinf[4];

#include "op_lib_cpp.h"

#include "airfoil_step7_1_save_soln_kernel.hpp"
#include "airfoil_step7_2_adt_calc_kernel.hpp"
#include "airfoil_step7_3_res_calc_kernel.hpp"
#include "airfoil_step7_4_bres_calc_kernel.hpp"
#include "airfoil_step7_5_update_kernel.hpp"
