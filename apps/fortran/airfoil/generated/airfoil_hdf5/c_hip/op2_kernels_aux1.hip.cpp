
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>

#include <cstdint>

extern double op2_const_gam;
extern double op2_const_gm1;
extern double op2_const_cfl;
extern double op2_const_eps;
extern double op2_const_mach;
extern double op2_const_alpha;
extern double op2_const_qinf[4];

static __constant__ double op2_const_gam_d;
static __constant__ double op2_const_gm1_d;
static __constant__ double op2_const_cfl_d;
static __constant__ double op2_const_eps_d;
static __constant__ double op2_const_mach_d;
static __constant__ double op2_const_alpha_d;
static __device__ double op2_const_qinf_d[4];

static uint64_t  op2_const_gam_hash = 0;
static uint64_t  op2_const_gm1_hash = 0;
static uint64_t  op2_const_cfl_hash = 0;
static uint64_t  op2_const_eps_hash = 0;
static uint64_t  op2_const_mach_hash = 0;
static uint64_t  op2_const_alpha_hash = 0;
static uint64_t  op2_const_qinf_hash = 0;

#define OP_F2C_PRELUDE OP_F2C_PRELUDE_1934688351
#define OP_F2C_PRELUDE_DATA OP_F2C_PRELUDE_1934688351_data
INCTXT(OP_F2C_PRELUDE, "op_f2c_prelude.h");

#include <op_f2c_prelude.h>
#include <op_f2c_helpers.h>

#include <op_lib_cpp.h>
#include <op_profile.h>

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

#include "airfoil_1_save_soln_kernel_aux1.hip.h"
#include "airfoil_2_adt_calc_kernel_aux1.hip.h"
#include "airfoil_3_res_calc_kernel_aux1.hip.h"
#include "airfoil_4_bres_calc_kernel_aux1.hip.h"
#include "airfoil_5_update_kernel_aux1.hip.h"
