#include "jac1_mpi_1_res_kernel_kernel.F90"
#include "jac1_mpi_2_update_kernel_kernel.F90"

module op2_kernels

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    use op2_m_jac1_mpi_1_res_kernel
    use op2_m_jac1_mpi_2_update_kernel

end module