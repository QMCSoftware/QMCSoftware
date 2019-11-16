""" Definition for CubLattice_g, a concrete implementation of StoppingCriterion """

from ._stopping_criterion import StoppingCriterion
from .._util import MaxSamplesWarning
from ..accum_data import MeanVarDataRep


class CubLattice_g(StoppingCriterion):
    """
    Stopping criterion for Lattice sequence with garunteed accuracy
    
    Guarantee
        This algorithm computes the integral of real valued functions in [0,1]^d
        with a prescribed generalized error tolerance. The Fourier coefficients
        of the integrand are assumed to be absolutely convergent. If the
        algorithm terminates without warning messages, the output is given with
        guarantees under the assumption that the integrand lies inside a cone of
        functions. The guarantee is based on the decay rate of the Fourier
        coefficients. For integration over domains other than [0,1]^d, this cone
        condition applies to f \circ \psi (the composition of the
        functions) where \psi is the transformation function for [0,1]^d to
        the desired region. For more details on how the cone is defined, please
        refer to the references below.
    """

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
            n_max (int): maximum number of samples
        """
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_max = n_max
        self.alpha = alpha
        self.inflate = inflate
        self.stage = "sigma"
        # Construct Data Object to House Integration data
        self.data = MeanVarData(len(true_measure), n_init)
        # Verify Compliant Construction
        allowed_distribs = ["Lattice"]
        super().__init__(discrete_distrib, allowed_distribs)

    def stop_yet(self):
        """ Determine when to stop """
        raise Exception("Not yet implemented")
