#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_aero_4_spMV {

inline void spMV(double *v0, double *v1, double *v2, double *v3, const double *K,
                 const double *p0, const double *p1, const double *p2, const double *p3) {
  //     double localsum = 0;
  //  for (int j=0; j<4; j++) {
  //         localsum = 0;
  //         for (int k = 0; k<4; k++) {
  //                 localsum += OP2_STRIDE(K, (j*4+k)] * p[k][0];
  //         }
  //         v[j][0] += localsum;
  //     }
  // }
  //
  //  for (int j=0; j<4; j++) {
  //    v[j][0] += OP2_STRIDE(K, (j*4+j)] * p[j][0];
  //         for (int k = j+1; k<4; k++) {
  //      double mult = OP2_STRIDE(K, (j*4+k)];
  //             v[j][0] += mult * p[k][0];
  //      v[k][0] += mult * p[j][0];
  //         }
  //     }
  // }
  v0[0] += K[0] * p0[0];
  v0[0] += K[1] * p1[0];
  v1[0] += K[1] * p0[0];
  v0[0] += K[2] * p2[0];
  v2[0] += K[2] * p0[0];
  v0[0] += K[3] * p3[0];
  v3[0] += K[3] * p0[0];
  v1[0] += K[4 + 1] * p1[0];
  v1[0] += K[4 + 2] * p2[0];
  v2[0] += K[4 + 2] * p1[0];
  v1[0] += K[4 + 3] * p3[0];
  v3[0] += K[4 + 3] * p1[0];
  v2[0] += K[8 + 2] * p2[0];
  v2[0] += K[8 + 3] * p3[0];
  v3[0] += K[8 + 3] * p2[0];
  v3[0] += K[15] * p3[0];
}}


void op_par_loop_aero_4_spMV(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2,
    op_arg arg3,
    op_arg arg4,
    op_arg arg5,
    op_arg arg6,
    op_arg arg7,
    op_arg arg8
) {
    int n_args = 9;
    op_arg args[9];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;
    args[3] = arg3;
    args[4] = arg4;
    args[5] = arg5;
    args[6] = arg6;
    args[7] = arg7;
    args[8] = arg8;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_aero_4_spMV::spMV(
            (double *)arg0.data + map0[0] * 1,
            (double *)arg1.data + map0[1] * 1,
            (double *)arg2.data + map0[2] * 1,
            (double *)arg3.data + map0[3] * 1,
            (double *)arg4.data + n * 16,
            (double *)arg5.data + map0[0] * 1,
            (double *)arg6.data + map0[1] * 1,
            (double *)arg7.data + map0[2] * 1,
            (double *)arg8.data + map0[3] * 1
        );

        if (n == set->size - 1) {
        }
    }

    if (n_exec < set->size) {
    }

    if (n_exec == 0 || n_exec == set->core_size)
        op_mpi_wait_all(n_args, args);


    op_mpi_set_dirtybit(n_args, args);
}