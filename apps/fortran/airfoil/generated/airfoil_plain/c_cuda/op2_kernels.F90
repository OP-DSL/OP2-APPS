#include "airfoil_1_save_soln_kernel.F90"
#include "airfoil_2_adt_calc_kernel.F90"
#include "airfoil_3_res_calc_kernel.F90"
#include "airfoil_4_bres_calc_kernel.F90"
#include "airfoil_5_update_kernel.F90"

module op2_kernels

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    use op2_m_airfoil_1_save_soln
    use op2_m_airfoil_2_adt_calc
    use op2_m_airfoil_3_res_calc
    use op2_m_airfoil_4_bres_calc
    use op2_m_airfoil_5_update

end module