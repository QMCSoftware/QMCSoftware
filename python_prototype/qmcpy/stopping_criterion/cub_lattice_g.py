""" Definition for CubLattice_g, a concrete implementation of StoppingCriterion """

from ._stopping_criterion import StoppingCriterion
from ..accum_data import CubatureData
from ..util import MaxSamplesWarning

from numpy import log2
import warnings

class CubLattice_g(StoppingCriterion):
    """
    Stopping Criterion quasi-Monte Carlo method using rank-1 Lattices cubature over
    a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Fourier coefficients cone decay assumptions.

    Guarantee
        This algorithm computes the integral of real valued functions in :math:`[0,1]^d`
        with a prescribed generalized error tolerance. The Fourier coefficients
        of the integrand are assumed to be absolutely convergent. If the
        algorithm terminates without warning messages, the output is given with
        guarantees under the assumption that the integrand lies inside a cone of
        functions. The guarantee is based on the decay rate of the Fourier
        coefficients. For integration over domains other than :math:`[0,1]^d`, this cone
        condition applies to :math:`f \circ \psi` (the composition of the
        functions) where :math:`\psi` is the transformation function for :math:`[0,1]^d` to
        the desired region. For more details on how the cone is defined, please
        refer to the references below.
    """

    def __init__(self, distrib, measure,
                 inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0,
                 n_init=2**10, n_max=2**35,
                 fudge = lambda m: 5*2**(-m)):
        """
        Args:
            distrib
            measure (Distribution): an instance of Distribution
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_max (int): maximum number of samples
            fudge (function): positive function multiplying the finite
                              sum of Fast Fourier coefficients specified 
                              in the cone of functions
        """
        # Input Checks
        levels = len(measure)
        if levels != 1:
            raise NotYetImplemented('''
                cub_lattice_g not implemented for multi-level problems.
                Use CLT stopping criterion with an iid distribution for multi-level problems ''')
        # Set Attributes
        self.inflate = inflate
        self.alpha = alpha
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        self.n_max = n_max
        self.stage = None
        m_min = log2(self.n_init)
        m_max = log2(self.n_max)
        if m_min%1 != 0 or m_max%1 != 0:
            warning_s = ' n_init and n_max must be a powers of 2. Using n_init = 2**10 and n_max=2**35'
            warnings.warn(warning_s, ParameterWarning)
            self.m_min = 10
            self.m_max = 35
        # Construct Data Object to House Integration data
        self.data = CubatureData(len(measure), m_min, m_max, fudge)
        # Verify Compliant Construction
        allowed_distribs = ["Lattice"]
        super().__init__(distrib, allowed_distribs)

    def stop_yet(self):
        """ Determine when to stop """
        # Check the end of the algorithm
        errest = self.data.fudge(self.data.m)*self.data.stilde
        ub = max(self.abs_tol, self.rel_tol*abs(self.data.solution + errest))
        lb = max(self.abs_tol, self.rel_tol*abs(self.data.solution - errest))
        self.data.solution = self.data.solution - errest*(ub-lb) / (ub+lb) # Optimal estimator
        if 4*errest**2/(ub+lb)**2 <= 1:
            self.stage = 'done'
        elif self.data.m == self.data.m_max:
            warning_s = """
            Alread generated %d samples.
            Trying to generate %s new samples, which exceeds n_max = %d.
            No more samples will be generated.
            Note that error tolerances may no longer be satisfied""" \
            % (int(self.data.n_total.sum()), str(self.data.n), int(self.n_max))
            warnings.warn(warning_s, MaxSamplesWarning)
            self.stage = 'done'
        else: # double sample size
            self.data.m += 1
                
