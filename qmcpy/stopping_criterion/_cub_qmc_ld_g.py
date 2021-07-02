from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import LDTransformData
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning
from numpy import *
from time import time
import warnings


class CubQMCLDG(StoppingCriterion):
    """
    Abstract class for CubQMC{LD}G where LD is a low discrepancy discrete distribution. 
    See subclasses for implementation differences for each LD sequence. 
    """

    def __init__(self, integrand, abs_tol, rel_tol, n_init, n_max, fudge, check_cone,
        control_variates, control_variate_means, update_beta, ptransform,
        coefv, allowed_levels, allowed_distribs, cast_complex):
        self.parameters = ['abs_tol','rel_tol','n_init','n_max']
        # Input Checks
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        m_min = log2(n_init)
        m_max = log2(n_max)
        if m_min%1 != 0. or m_min < 8. or m_max%1 != 0.:
            warning_s = '''
                n_init and n_max must be a powers of 2.
                n_init must be >= 2^8.
                Using n_init = 2^10 and n_max=2^35.'''
            warnings.warn(warning_s, ParameterWarning)
            m_min = 10.
            m_max = 35.
        self.n_init = 2.**m_min
        self.n_max = 2.**m_max
        self.m_min = m_min
        self.m_max = m_max
        self.fudge = fudge
        self.check_cone = check_cone
        self.coefv = coefv
        self.ptransform = ptransform
        self.cast_complex = cast_complex
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.cv = control_variates
        self.cv_mu = control_variate_means
        self.ub = update_beta
        # Verify Compliant Construction
        super(CubQMCLDG,self).__init__(allowed_levels, allowed_distribs)

    def integrate(self):
        """ See abstract method. """
        # Construct AccumulateData Object to House Integration data
        self.data = LDTransformData(self, self.integrand, self.true_measure, self.discrete_distrib,
            self.coefv, self.m_min, self.m_max, self.fudge, self.check_cone, ptransform=self.ptransform, 
            cast_complex=self.cast_complex, control_variates=self.cv, control_variate_means=self.cv_mu, update_beta=self.ub)
        t_start = time()
        while True:
            self.data.update_data()
            # Check the end of the algorithm
            self.data.error_bound = self.data.fudge(self.data.m)*self.data.stilde
            # Compute optimal estimator
            ub = max(self.abs_tol, self.rel_tol*abs(self.data.solution + self.data.error_bound))
            lb = max(self.abs_tol, self.rel_tol*abs(self.data.solution - self.data.error_bound))
            self.data.solution = self.data.solution - self.data.error_bound*(ub-lb) / (ub+lb)
            if 4*self.data.error_bound**2./(ub+lb)**2. <= 1.:
                # stopping criterion met
                break
            elif self.data.m == self.data.m_max:
                # doubling samples would go over n_max
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples would exceed n_max = %d.
                No more samples will be generated.
                Note that error tolerances may no longer be satisfied""" \
                % (int(2**self.data.m), int(2**self.data.m), int(2**self.data.m_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                # double sample size
                self.data.m += 1.
        self.data.time_integrate = time() - t_start
        return self.data.solution, self.data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol != None: self.abs_tol = abs_tol
        if rel_tol != None: self.rel_tol = rel_tol
