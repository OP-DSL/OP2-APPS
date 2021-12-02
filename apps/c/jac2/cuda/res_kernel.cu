//
// auto-generated by op2.py
//

//user function
__device__ void res_gpu( const double *A, const float *u, float *du, const float *beta) {
  *du += (float)((*beta) * (*A) * (*u));

}

// CUDA kernel function
__global__ void op_cuda_res(
  const float *__restrict ind_arg0,
  float *__restrict ind_arg1,
  const int *__restrict opDat1Map,
  const double *__restrict arg0,
  const float *arg3,
  int start,
  int end,
  int   set_size) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid + start < end) {
    int n = tid + start;
    //initialise local variables
    float arg2_l[3];
    for ( int d=0; d<3; d++ ){
      arg2_l[d] = ZERO_float;
    }
    int map1idx;
    int map2idx;
    map1idx = opDat1Map[n + set_size * 1];
    map2idx = opDat1Map[n + set_size * 0];

    //user-supplied kernel call
    res_gpu(arg0+n*3,
        ind_arg0+map1idx*2,
        arg2_l,
        arg3);
    atomicAdd(&ind_arg1[0+map2idx*3],arg2_l[0]);
    atomicAdd(&ind_arg1[1+map2idx*3],arg2_l[1]);
    atomicAdd(&ind_arg1[2+map2idx*3],arg2_l[2]);
  }
}


//host stub function
void op_par_loop_res(char const *name, op_set set,
  op_arg arg0,
  op_arg arg1,
  op_arg arg2,
  op_arg arg3){

  float*arg3h = (float *)arg3.data;
  int nargs = 4;
  op_arg args[4];

  args[0] = arg0;
  args[1] = arg1;
  args[2] = arg2;
  args[3] = arg3;

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(0);
  op_timers_core(&cpu_t1, &wall_t1);
  OP_kernels[0].name      = name;
  OP_kernels[0].count    += 1;


  int    ninds   = 2;
  int    inds[4] = {-1,0,1,-1};

  if (OP_diags>2) {
    printf(" kernel routine with indirection: res\n");
  }
  int set_size = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);
  if (set_size > 0) {

    //transfer constants to GPU
    int consts_bytes = 0;
    consts_bytes += ROUND_UP(1*sizeof(float));
    reallocConstArrays(consts_bytes);
    consts_bytes = 0;
    arg3.data   = OP_consts_h + consts_bytes;
    arg3.data_d = OP_consts_d + consts_bytes;
    for ( int d=0; d<1; d++ ){
      ((float *)arg3.data)[d] = arg3h[d];
    }
    consts_bytes += ROUND_UP(1*sizeof(float));
    mvConstArraysToDevice(consts_bytes);

    //set CUDA execution parameters
    #ifdef OP_BLOCK_SIZE_0
      int nthread = OP_BLOCK_SIZE_0;
    #else
      int nthread = OP_block_size;
    #endif

    for ( int round=0; round<2; round++ ){
      if (round==1) {
        op_mpi_wait_all_grouped(nargs, args, 2);
      }
      int start = round==0 ? 0 : set->core_size;
      int end = round==0 ? set->core_size : set->size + set->exec_size;
      if (end-start>0) {
        int nblocks = (end-start-1)/nthread+1;
        op_cuda_res<<<nblocks,nthread>>>(
        (float *)arg1.data_d,
        (float *)arg2.data_d,
        arg1.map_data_d,
        (double*)arg0.data_d,
        (float*)arg3.data_d,
        start,end,set->size+set->exec_size);
      }
    }
  }
  op_mpi_set_dirtybit_cuda(nargs, args);
  cutilSafeCall(cudaDeviceSynchronize());
  //update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[0].time     += wall_t2 - wall_t1;
}