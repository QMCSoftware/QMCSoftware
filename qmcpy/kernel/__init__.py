from .common_kernels import (
    KernelGaussian,
    KernelSquaredExponential,
    KernelRationalQuadratic,
    KernelMatern12,
    KernelMatern32,
    KernelMatern52,
)
from .si_dsi_kernels import (
    KernelShiftInvar,
    KernelShiftInvarCombined,
    KernelDigShiftInvar,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDigShiftInvarCombined,
)
from .multitask_kernel import KernelMultiTask, KernelMultiTaskDerivs

KernelSI = KernelShiftInvar
KernelSICombined = KernelShiftInvarCombined
KernelDSI = KernelDigShiftInvar
KernelDSIAA = KernelDigShiftInvarAdaptiveAlpha
KernelDSICombined = KernelDigShiftInvarCombined
