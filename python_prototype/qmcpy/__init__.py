""" Meta-data and public utilities for qmcpy """

name = "qmcpy"
__version__ = 0.1

# For making short import statements
from .integrate import integrate
from ._util import DistributionCompatibilityError, univ_repr, summarize
from .accum_data.mean_var_data import MeanVarData
from .accum_data.mean_var_data_rep import MeanVarDataRep
from .distribution.iid_distribution import IIDDistribution
from .distribution.quasi_random import QuasiRandom
from .integrand.asian_call import AsianCall
from .integrand.keister import Keister
from .integrand.linear import Linear
from .measures.measures import *
from .stop.clt import CLT
from .stop.clt_rep import CLTRep

