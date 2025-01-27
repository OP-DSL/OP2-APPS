#define UNUSED(x) if (.false.) print *, SHAPE(x)

module op2_consts

    use cudafor

    implicit none

    real(8) :: op2_const_alpha
    real(8), constant :: op2_const_alpha_d

contains

    subroutine op_decl_const_alpha(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)

        op2_const_alpha = ptr
    end subroutine

    subroutine op_update_const_cuda_alpha()
        op2_const_alpha_d = op2_const_alpha
    end subroutine

end module