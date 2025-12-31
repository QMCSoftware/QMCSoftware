from .abstract_discrete_distribution import AbstractDiscreteDistribution
from .iid_std_uniform import IIDStdUniform
from .lattice import Lattice
from .digital_net_b2 import DigitalNetB2
from .digital_net_any_base import DigitalNetAnyBases,Halton,Faure

DiscreteDistribution = AbstractDiscreteDistribution
_DiscreteDistribution = AbstractDiscreteDistribution
Sobol = DigitalNetB2
DigitalNet = DigitalNetB2
Net = DigitalNetB2
NetB2 = DigitalNetB2

