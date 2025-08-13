#include <op_lib_cpp.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace op2_m_aero_mpi_4_spMV {

inline void spMV(double **v, const double *K, const double **p) {
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
  v[0][0] += K[0] * p[0][0];
  v[0][0] += K[1] * p[1][0];
  v[1][0] += K[1] * p[0][0];
  v[0][0] += K[2] * p[2][0];
  v[2][0] += K[2] * p[0][0];
  v[0][0] += K[3] * p[3][0];
  v[3][0] += K[3] * p[0][0];
  v[1][0] += K[4 + 1] * p[1][0];
  v[1][0] += K[4 + 2] * p[2][0];
  v[2][0] += K[4 + 2] * p[1][0];
  v[1][0] += K[4 + 3] * p[3][0];
  v[3][0] += K[4 + 3] * p[1][0];
  v[2][0] += K[8 + 2] * p[2][0];
  v[2][0] += K[8 + 3] * p[3][0];
  v[3][0] += K[8 + 3] * p[2][0];
  v[3][0] += K[15] * p[3][0];
}}


void op_par_loop_aero_mpi_4_spMV(
    const char* name,
    op_set set,
    op_arg arg0,
    op_arg arg1,
    op_arg arg2
) {
    int n_args = 3;
    op_arg args[3];

    args[0] = arg0;
    args[1] = arg1;
    args[2] = arg2;

    int n_exec = op_mpi_halo_exchanges(set, n_args, args);



    for (int n = 0; n < n_exec; ++n) {
        if (n == set->core_size) {
            op_mpi_wait_all(n_args, args);
        }

        int *map0 = arg0.map_data + n * arg0.map->dim;


        op2_m_aero_mpi_4_spMV::spMV(
            (double *)arg0.data + map0[-4] * 1,
            (double *)arg1.data + n * 16,
            (double *)arg2.data + map0[-4] * 1
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