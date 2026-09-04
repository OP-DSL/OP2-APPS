#define UNUSED(x) if (.false.) print *, SHAPE(x)

#include "jac_1_res_kernel.F90"
#include "jac_2_update_kernel.F90"

module op2_kernels

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    use op2_m_jac_1_res
    use op2_m_jac_2_update

end module