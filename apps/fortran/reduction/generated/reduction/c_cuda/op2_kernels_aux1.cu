
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>

#include <cstdint>




#define OP_F2C_PRELUDE OP_F2C_PRELUDE_2995257982
#define OP_F2C_PRELUDE_DATA OP_F2C_PRELUDE_2995257982_data
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

#include "reduction_1_cell_count_kernel_aux1.cuh"
#include "reduction_2_edge_count_kernel_aux1.cuh"
