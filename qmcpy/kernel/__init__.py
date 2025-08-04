from .common_kernels import (
    KernelGaussian,
    KernelSquaredExponential,
    KernelRationalQuadratic,
    KernelMatern12,
    KernelMatern32,
    KernelMatern52,
)
from .si_dsi_kernels import KernelShiftInvar,KernelDigShiftInvar
from .multitask_kernel import KernelMultiTask,KernelMultiTaskDerivs

KernelSI = KernelShiftInvar
KernelDSI = KernelDigShiftInvar