#define UNUSED(x) if (.false.) print *, SHAPE(x)

module op2_consts

    implicit none

    real(8) :: op2_const_gam
    real(8) :: op2_const_gm1
    real(8) :: op2_const_cfl
    real(8) :: op2_const_eps
    real(8) :: op2_const_mach
    real(8) :: op2_const_alpha
    real(8), dimension(4) :: op2_const_qinf

contains

    subroutine op_decl_const_gam(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)
        op2_const_gam = ptr
    end subroutine

    subroutine op_decl_const_gm1(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)
        op2_const_gm1 = ptr
    end subroutine

    subroutine op_decl_const_cfl(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)
        op2_const_cfl = ptr
    end subroutine

    subroutine op_decl_const_eps(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)
        op2_const_eps = ptr
    end subroutine

    subroutine op_decl_const_mach(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)
        op2_const_mach = ptr
    end subroutine

    subroutine op_decl_const_alpha(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)
        op2_const_alpha = ptr
    end subroutine

    subroutine op_decl_const_qinf(ptr, dim)
        real(8), dimension(4) :: ptr
        integer(4) :: dim

        integer(4) :: d

        do d = 1, dim
            op2_const_qinf(d) = ptr(d)
        end do
    end subroutine

end module