#include <cstdint>

extern double alpha;

static __constant__ double op2_const_alpha_d;

static uint64_t  op2_const_alpha_hash = 0;

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>

// Note: OP_F2C_PARAMS unused in C++ backend (can be simply extended if needed)
// #define OP_F2C_PARAMS OP_F2C_PARAMS_9124810794
// #define OP_F2C_PARAMS_DATA OP_F2C_PARAMS_9124810794_data
// INCTXT(OP_F2C_PARAMS, "op2_const_list_params.h");

#define OP_F2C_PRELUDE OP_F2C_PRELUDE_9124810794
#define OP_F2C_PRELUDE_DATA OP_F2C_PRELUDE_9124810794_data
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

#include "jac_1_res_kernel.h"
#include "jac_2_update_kernel.h"
