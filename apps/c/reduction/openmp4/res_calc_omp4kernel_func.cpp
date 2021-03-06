//
// auto-generated by op2.py
//

void res_calc_omp4_kernel(
  int *map0,
  int map0size,
  int *arg1,
  double *data0,
  int dat0size,
  int *col_reord,
  int set_size1,
  int start,
  int end,
  int num_teams,
  int nthread){

  int arg1_l = *arg1;
  #pragma omp target teams num_teams(num_teams) thread_limit(nthread) \
    map(to:col_reord[0:set_size1],map0[0:map0size],data0[0:dat0size])\
    map(tofrom: arg1_l) reduction(+:arg1_l)
  #pragma omp distribute parallel for schedule(static,1) reduction(+:arg1_l)
  for ( int e=start; e<end; e++ ){
    int n_op = col_reord[e];
    int map0idx;
    map0idx = map0[n_op + set_size1 * 0];

    //variable mapping
    double *data = &data0[4 * map0idx];
    int *count = &arg1_l;

    //inline function
    
    data[0] = 0.0;
    (*count)++;
    //end inline func
  }

  *arg1 = arg1_l;
}
