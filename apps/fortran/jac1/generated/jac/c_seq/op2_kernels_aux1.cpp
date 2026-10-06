extern double op2_const_alpha;

#include <op_f2c_prelude.h>
#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

#include "jac_1_res_kernel_aux1.hpp"
#include "jac_2_update_kernel_aux1.hpp"
