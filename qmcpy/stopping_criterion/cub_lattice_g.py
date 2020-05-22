""" Definition for CubLattice_g, a concrete implementation of StoppingCriterion

Adapted from 
    https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/cubLattice_g.m

Reference:
    
    [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
    Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
    GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
    Available from http://gailgithub.github.io/GAIL_Dev/
"""

from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import CubatureData
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning
from numpy import log2, hstack, tile, exp, pi, arange
from time import perf_counter
import warnings


class CubLattice_g(StoppingCriterion):
    """
    Stopping Criterion quasi-Monte Carlo method using rank-1 Lattices cubature over
    a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Fourier coefficients cone decay assumptions.

    Guarantee
        This algorithm computes the integral of real valued functions in $[0,1]^d$
        with a prescribed generalized error tolerance. The Fourier coefficients
        of the integrand are assumed to be absolutely convergent. If the
        algorithm terminates without warning messages, the output is given with
        guarantees under the assumption that the integrand lies inside a cone of
        functions. The guarantee is based on the decay rate of the Fourier
        coefficients. For integration over domains other than $[0,1]^d$, this cone
        condition applies to $f \circ \psi$ (the composition of the
        functions) where $\psi$ is the transformation function for $[0,1]^d$ to
        the desired region. For more details on how the cone is defined, please
        refer to the references below.
    """

    parameters = ['abs_tol','rel_tol','n_init','n_max']

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0, n_init=2**10, n_max=2**35,
                 fudge=lambda m: 5*2**(-m), check_cone=False):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            abs_tol (float): absolute error tolerance
            rel_tol (float): relative error tolerance
            n_init (int): initial number of samples
            n_max (int): maximum number of samples
            fudge (function): positive function multiplying the finite
                              sum of Fast Fourier coefficients specified 
                              in the cone of functions
            check_cone (boolean): check if the function falls in the cone
            """
        # Input Checks
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        m_min = log2(n_init)
        m_max = log2(n_max)
        if m_min%1 != 0 or m_min < 8 or m_max%1 != 0:
            warning_s = '''
                n_init and n_max must be a powers of 2.
                n_init must be >= 2^8.
                Using n_init = 2^10 and n_max=2^35.'''
            warnings.warn(warning_s, ParameterWarning)
            m_min = 10
            m_max = 35
        self.n_init = 2**m_min
        self.n_max = 2**m_max
        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        allowed_levels = 'single'
        allowed_distribs = ["Lattice"]
        super().__init__(distribution, allowed_levels, allowed_distribs)
        if not distribution.scramble:
            raise ParameterError("CubLattice_g requires distribution to have scramble=True")
        if distribution.backend != 'gail':
            raise ParameterError("CubLattice_g requires distribution to have 'GAIL' backend")
        # Construct AccumulateData Object to House Integration data
        self.data = CubatureData(self, integrand, self.fft_update, m_min, m_max, fudge, check_cone)

    def integrate(self):
        """ Determine when to stop """
        t_start = perf_counter()
        while True:
            self.data.update_data()
            # Check the end of the algorithm
            errest = self.data.fudge(self.data.m)*self.data.stilde
            # Compute optimal estimator
            ub = max(self.abs_tol, self.rel_tol*abs(self.data.solution + errest))
            lb = max(self.abs_tol, self.rel_tol*abs(self.data.solution - errest))
            self.data.solution = self.data.solution - errest*(ub-lb) / (ub+lb)
            if 4*errest**2/(ub+lb)**2 <= 1:
                # stopping criterion met
                break
            elif self.data.m == self.data.m_max:
                # doubling samples would go over n_max
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples would exceed n_max = %d.
                No more samples will be generated.
                Note that error tolerances may no longer be satisfied""" \
                % (int(2**self.data.m), int(self.data.m), int(2**self.data.m_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                # double sample size
                self.data.m += 1
        self.data.time_integrate = perf_counter() - t_start
        return self.data.solution, self.data
            
    def fft_update(self, y, ynext):
        """
        Fast Fourier Transform (FFT) ynext, combine with y, then FFT all points
        Args:
            y (ndarray): all previous samples
            ynext (ndarray): next samples
        Return:
            y (ndarray): all points combined and transformed
        """
        y = y.astype(complex)
        ynext = ynext.astype(complex)
        ## Compute initial FFT on next points
        mnext = int(log2(len(ynext)))
        for l in range(mnext):
            nl = 2**l
            nmminlm1 = 2**(mnext-l-1)
            ptind_nl = hstack(( tile(True,nl), tile(False,nl) ))
            ptind = tile(ptind_nl,int(nmminlm1))
            coef = exp(-2*pi*1j*arange(nl)/(2*nl))
            coefv = tile(coef,int(nmminlm1))
            evenval = ynext[ptind]
            oddval = ynext[~ptind]
            ynext[ptind] = (evenval + coefv*oddval) / 2
            ynext[~ptind] = (evenval - coefv*oddval) / 2
        y = hstack((y,ynext))
        if len(y) > len(ynext): # already generated some samples samples
            ## Compute FFT on all points
            nl = 2**mnext
            ptind = hstack((tile(True,int(nl)),tile(False,int(nl))))
            coefv = exp(-2*pi*1j*arange(nl)/(2*nl))
            evenval = y[ptind]
            oddval = y[~ptind]
            y[ptind] = (evenval + coefv*oddval) /2
            y[~ptind] = (evenval - coefv*oddval) /2
        return y
