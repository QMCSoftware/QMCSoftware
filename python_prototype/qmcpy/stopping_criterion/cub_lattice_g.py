""" Definition for CubLattice_g, a concrete implementation of StoppingCriterion """

from ._stopping_criterion import StoppingCriterion
from .._util import MaxSamplesWarning
from ..accum_data import MeanVarDataRep


class CubLattice_g(StoppingCriterion):
    """ Stopping criterion for Lattice sequence with garunteed accuracy """

    def __init__(self, discrete_distrib, true_measure,
                 replications=16, inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0,
                 n_init=1024, n_max=2**20):
        """
        Args:
            discrete_distrib
            true_measure (DiscreteDistribution): an instance of DiscreteDistribution
            replications (int): number of random nxm matrices to generate
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
        """
        allowed_distribs = ["Sobol"]  # supported distributions
        super().__init__(discrete_distrib, allowed_distribs, abs_tol,
                         rel_tol, n_init, n_max)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha  # uncertainty level
        self.stage = "begin"
        # Construct Data Object
        n_integrands = len(true_measure)
        self.data = MeanVarDataRep(n_integrands, replications)
        # house integration data
        self.data.n = tile(self.n_init, n_integrands)  # next n for each integrand

    def stop_yet(self):
        """ Determine when to stop """
        raise Exception("Not yet implemented")
