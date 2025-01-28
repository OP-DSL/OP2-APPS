#include "hydra_const_list_c_cuda.cuh"

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>


#define OP_F2C_PRELUDE OP_F2C_PRELUDE_5730915850
#define OP_F2C_PARAMS OP_F2C_PARAMS_5730915850

#define OP_F2C_PRELUDE_DATA OP_F2C_PRELUDE_5730915850_data
#define OP_F2C_PARAMS_DATA OP_F2C_PARAMS_5730915850_data

INCTXT(OP_F2C_PRELUDE, "op_f2c_prelude.h");
INCTXT(OP_F2C_PARAMS, "hydra_const_list_params.h");

#include <op_f2c_prelude.h>
#include <op_f2c_helpers.h>

#include <op_lib_cpp.h>
#include <op_timing2.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

extern "C" {

void prepareDeviceGbls(op_arg *args, int nargs, int max_threads);
bool processDeviceGbls(op_arg *args, int nargs, int nthreads, int max_threads);
int getBlockLimit(op_arg *args, int nargs, int block_size, const char *name);
void setGblIncAtomic(bool enable);

}

#include "airfoil_1_save_soln_kernel_aux1.cuh"
#include "airfoil_2_adt_calc_kernel_aux1.cuh"
#include "airfoil_3_res_calc_kernel_aux1.cuh"
#include "airfoil_4_bres_calc_kernel_aux1.cuh"
#include "airfoil_5_update_kernel_aux1.cuh"
