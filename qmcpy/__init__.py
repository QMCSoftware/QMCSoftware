from .discrete_distribution import *
from .true_measure import *
from .integrand import *
from .stopping_criterion import *
from .kernel import (
    KernelGaussian,
    KernelSquaredExponential,
    KernelRationalQuadratic,
    KernelMatern12,
    KernelMatern32,
    KernelMatern52,
    KernelShiftInvar,
    KernelSI,
    KernelShiftInvarCombined,
    KernelSICombined,
    KernelDigShiftInvar,
    KernelDSI,
    KernelDigShiftInvarAdaptiveAlpha,
    KernelDSIAA,
    KernelDigShiftInvarCombined,
    KernelDSICombined,
    KernelMultiTask,
    KernelMultiTaskDerivs,
)
from .true_measure.acceptance_rejection import (
    AcceptanceRejection,
    AcceptanceRejectionReal,
)
from .true_measure.triangular import TriangularDistribution
from .fast_transform import (
    fftbr,
    ifftbr,
    fwht,
    omega_fftbr,
    omega_fwht,
    fftbr_torch,
    ifftbr_torch,
    fwht_torch,
    omega_fftbr_torch,
    omega_fwht_torch,
)
from .util import (
    CubatureWarning,
    DimensionError,
    DistributionCompatibilityError,
    ExactGPyTorchRegressionModel,
    MaxLevelsWarning,
    MaxSamplesWarning,
    MethodImplementationError,
    NotYetImplemented,
    ParameterError,
    ParameterWarning,
    latnetbuilder_linker,
    mlmc_test,
    plot_proj,
    stop_notebook,
)
from .util.data import Data
from .util.dig_shift_invar_ops import to_bin, to_float, weighted_walsh_funcs
from .util.shift_invar_ops import bernoulli_poly
from .util.torch_numpy_ops import get_npt
from .util.transforms import (
    insert_batch_dims,
    parse_assign_param,
    tf_exp,
    tf_exp_eps,
    tf_exp_eps_inv,
    tf_exp_inv,
    tf_explinear,
    tf_explinear_inv,
    tf_identity,
    tf_square,
    tf_square_inv,
)

try:
    _mpmc_utils_available = False
    # Keep the Torch-only utilities available when the heavier PyG import fails.
    from .discrete_distribution.mpmc import utils as mpmc_utils

    _mpmc_utils_available = True
    from .discrete_distribution.mpmc.models import MPMC_net
except ImportError as error:
    _missing_module = getattr(error, "name", None)
    if _missing_module is None or _missing_module.split(".", 1)[0] not in {
        "pyg_lib",
        "torch",
        "torch_geometric",
    }:
        raise

    def _raise_missing_mpmc_dependency(component):
        raise ModuleNotFoundError(
            f"{component} requires optional MPMC dependencies; missing module "
            f"'{_missing_module}'. Install torch, pyg_lib, and torch-geometric.",
            name=_missing_module,
        )

    if not _mpmc_utils_available:
        class _MissingMPMCUtils(object):
            def __getattr__(self, _name):
                _raise_missing_mpmc_dependency("mpmc_utils")

        mpmc_utils = _MissingMPMCUtils()

    class MPMC_net(object):
        def __init__(self, *args, **kwargs):
            _raise_missing_mpmc_dependency("MPMC_net")

name = "qmcpy"
__version__ = "2.3"
