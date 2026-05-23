#include <cstdint>

extern double gam;
extern double gm1;
extern double gm1i;
extern double m2;
extern double wtg1[2];
extern double xi1[2];
extern double Ng1[4];
extern double Ng1_xi[4];
extern double wtg2[4];
extern double Ng2[16];
extern double Ng2_xi[32];
extern double minf;
extern double freq;
extern double kappa;
extern double nmode;
extern double mfan;

static __constant__ double op2_const_gam_d;
static __constant__ double op2_const_gm1_d;
static __constant__ double op2_const_gm1i_d;
static __constant__ double op2_const_m2_d;
static __device__ double op2_const_wtg1_d[2];
static __device__ double op2_const_xi1_d[2];
static __device__ double op2_const_Ng1_d[4];
static __device__ double op2_const_Ng1_xi_d[4];
static __device__ double op2_const_wtg2_d[4];
static __device__ double op2_const_Ng2_d[16];
static __device__ double op2_const_Ng2_xi_d[32];
static __constant__ double op2_const_minf_d;
static __constant__ double op2_const_freq_d;
static __constant__ double op2_const_kappa_d;
static __constant__ double op2_const_nmode_d;
static __constant__ double op2_const_mfan_d;

static uint64_t  op2_const_gam_hash = 0;
static uint64_t  op2_const_gm1_hash = 0;
static uint64_t  op2_const_gm1i_hash = 0;
static uint64_t  op2_const_m2_hash = 0;
static uint64_t  op2_const_wtg1_hash = 0;
static uint64_t  op2_const_xi1_hash = 0;
static uint64_t  op2_const_Ng1_hash = 0;
static uint64_t  op2_const_Ng1_xi_hash = 0;
static uint64_t  op2_const_wtg2_hash = 0;
static uint64_t  op2_const_Ng2_hash = 0;
static uint64_t  op2_const_Ng2_xi_hash = 0;
static uint64_t  op2_const_minf_hash = 0;
static uint64_t  op2_const_freq_hash = 0;
static uint64_t  op2_const_kappa_hash = 0;
static uint64_t  op2_const_nmode_hash = 0;
static uint64_t  op2_const_mfan_hash = 0;

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>

// Note: OP_F2C_PARAMS unused in C++ backend (can be simply extended if needed)
// #define OP_F2C_PARAMS OP_F2C_PARAMS_3975597720
// #define OP_F2C_PARAMS_DATA OP_F2C_PARAMS_3975597720_data
// INCTXT(OP_F2C_PARAMS, "op2_const_list_params.h");

#define OP_F2C_PRELUDE OP_F2C_PRELUDE_3975597720
#define OP_F2C_PRELUDE_DATA OP_F2C_PRELUDE_3975597720_data
INCTXT(OP_F2C_PRELUDE, "op_f2c_prelude.h");


#include <op_f2c_prelude.h>
#include <op_f2c_helpers.h>

#include <op_lib_cpp.h>
#include <op_profile.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace f2c = op::f2c;

void op_decl_const_char(int dim, const char *type, int size, char *dat, const char *name) {
    // Unused with JIT
}

extern "C" {

void prepareDeviceGbls(op_arg *args, int nargs, int max_threads);
bool processDeviceGbls(op_arg *args, int nargs, int nthreads, int max_threads);
int getBlockLimit(op_arg *args, int nargs, int block_size, const char *name);
void setGblIncAtomic(bool enable);

}

#include "aero_1_res_calc_kernel.h"
#include "aero_2_dirichlet_kernel.h"
#include "aero_3_init_cg_kernel.h"
#include "aero_4_spMV_kernel.h"
#include "aero_5_dirichlet_kernel.h"
#include "aero_6_dotPV_kernel.h"
#include "aero_7_updateUR_kernel.h"
#include "aero_8_dotR_kernel.h"
#include "aero_9_updateP_kernel.h"
#include "aero_10_update_kernel.h"
