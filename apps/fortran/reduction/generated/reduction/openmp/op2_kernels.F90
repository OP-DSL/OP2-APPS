#define UNUSED(x) if (.false.) print *, SHAPE(x)

#include "reduction_1_cell_count_kernel.F90"
#include "reduction_2_edge_count_kernel.F90"

module op2_kernels

    use iso_c_binding

    use op2_fortran_declarations
    use op2_fortran_rt_support

    use op2_consts

    use op2_m_reduction_1_cell_count
    use op2_m_reduction_2_edge_count

end module