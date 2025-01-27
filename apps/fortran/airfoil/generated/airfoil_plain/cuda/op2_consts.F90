#define UNUSED(x) if (.false.) print *, SHAPE(x)

module op2_consts

    use cudafor

    implicit none

    real(8) :: op2_const_gam
    real(8), constant :: op2_const_gam_d

    real(8) :: op2_const_gm1
    real(8), constant :: op2_const_gm1_d

    real(8) :: op2_const_cfl
    real(8), constant :: op2_const_cfl_d

    real(8) :: op2_const_eps
    real(8), constant :: op2_const_eps_d

    real(8) :: op2_const_mach
    real(8), constant :: op2_const_mach_d

    real(8) :: op2_const_alpha
    real(8), constant :: op2_const_alpha_d

    real(8), dimension(4) :: op2_const_qinf
    real(8), constant, dimension(4) :: op2_const_qinf_d

contains

    subroutine op_decl_const_gam(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)

        op2_const_gam = ptr
    end subroutine

    subroutine op_update_const_cuda_gam()
        op2_const_gam_d = op2_const_gam
    end subroutine

    subroutine op_decl_const_gm1(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)

        op2_const_gm1 = ptr
    end subroutine

    subroutine op_update_const_cuda_gm1()
        op2_const_gm1_d = op2_const_gm1
    end subroutine

    subroutine op_decl_const_cfl(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)

        op2_const_cfl = ptr
    end subroutine

    subroutine op_update_const_cuda_cfl()
        op2_const_cfl_d = op2_const_cfl
    end subroutine

    subroutine op_decl_const_eps(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)

        op2_const_eps = ptr
    end subroutine

    subroutine op_update_const_cuda_eps()
        op2_const_eps_d = op2_const_eps
    end subroutine

    subroutine op_decl_const_mach(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)

        op2_const_mach = ptr
    end subroutine

    subroutine op_update_const_cuda_mach()
        op2_const_mach_d = op2_const_mach
    end subroutine

    subroutine op_decl_const_alpha(ptr, dim)
        real(8) :: ptr
        integer(4) :: dim

        UNUSED(dim)

        op2_const_alpha = ptr
    end subroutine

    subroutine op_update_const_cuda_alpha()
        op2_const_alpha_d = op2_const_alpha
    end subroutine

    subroutine op_decl_const_qinf(ptr, dim)
        real(8), dimension(4) :: ptr
        integer(4) :: dim

        integer(4) :: d

        do d = 1, dim
            op2_const_qinf(d) = ptr(d)
        end do
    end subroutine

    subroutine op_update_const_cuda_qinf()
        op2_const_qinf_d = op2_const_qinf
    end subroutine

end module