extern double op2_const_gam;
extern double op2_const_gm1;
extern double op2_const_cfl;
extern double op2_const_eps;
extern double op2_const_mach;
extern double op2_const_alpha;
extern double op2_const_qinf[4];

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

#include "airfoil_1_save_soln_kernel_aux1.hpp"
#include "airfoil_2_adt_calc_kernel_aux1.hpp"
#include "airfoil_3_res_calc_kernel_aux1.hpp"
#include "airfoil_4_bres_calc_kernel_aux1.hpp"
#include "airfoil_5_update_kernel_aux1.hpp"
