extern float gam;
extern float gm1;
extern float cfl;
extern float eps;
extern float mach;
extern float alpha;
extern float qinf[4];

#include "op_lib_cpp.h"

#include "airfoil_1_save_soln_kernel.hpp"
#include "airfoil_2_adt_calc_kernel.hpp"
#include "airfoil_3_res_calc_kernel.hpp"
#include "airfoil_4_bres_calc_kernel.hpp"
#include "airfoil_5_update_kernel.hpp"
