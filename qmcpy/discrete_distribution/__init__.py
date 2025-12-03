from .abstract_discrete_distribution import AbstractDiscreteDistribution
from .iid_std_uniform import IIDStdUniform
from .lattice import Lattice
from .digital_net_b2 import DigitalNetB2
from .halton import Halton

DiscreteDistribution = AbstractDiscreteDistribution
_DiscreteDistribution = AbstractDiscreteDistribution
Sobol = DigitalNetB2
DigitalNet = DigitalNetB2
Net = DigitalNetB2

# Import MPMC if PyTorch dependencies are available
try:
    from .mpmc import MPMC
except Exception:  # pragma: no cover - optional dependency
    MPMC = None


