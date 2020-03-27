"""
Definition for MeanMC_g, a concrete implementation of StoppingCriterion

Adapted from
    https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/meanMC_g.m

Reference:
    
    [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
    Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
    GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
    Available from http://gailgithub.github.io/GAIL_Dev/
"""

from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarData
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..util import MaxSamplesWarning, NotYetImplemented
from numpy import array, ceil, exp, floor, log, minimum, sqrt, tile
from scipy.optimize import fsolve
from scipy.stats import norm
from time import process_time
import warnings


class MeanMC_g(StoppingCriterion):
    """
    Stopping Criterion with garunteed accuracy

    Guarantee
        This algorithm attempts to calculate the mean, mu, of a random variable
        to a prescribed error tolerance, tolfun:= max(abstol,reltol*|mu|), with
        guaranteed confidence level 1-alpha. If the algorithm terminates without
        showing any warning messages and provides an answer tmu, then the follow
        inequality would be satisfied:
            Pr(|mu-tmu| <= tolfun) >= 1-alpha
    """

    parameters = ['inflate','alpha','abs_tol','rel_tol','n_init','n_max']
    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0, n_init=1024, n_max=1e10,
                 inflate=1.2, alpha=0.01):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate: inflation factor when estimating variance
            alpha: significance level for confidence interval
            abs_tol: absolute error tolerance
            rel_tol: relative error tolerance
            n_init: initial number of samples
            n_max: maximum number of samples
        """
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        self.n_max = n_max
        self.alpha = alpha
        self.inflate = inflate
        self.alpha_sigma = self.alpha / 2  # the uncertainty for variance estimation
        self.kurtmax = (n_init - 3) / (n_init - 1) + \
            (self.alpha_sigma * n_init) / (1 - self.alpha_sigma) * \
            (1 - 1 / self.inflate**2)**2
        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        allowed_levels = 'single'
        allowed_distribs = ["IIDStdUniform", "IIDStdGaussian"]
        super().__init__(distribution, allowed_levels, allowed_distribs)
        # Construct AccumulateData Object to House Integration data
        self.data = MeanVarData(self, integrand, self.n_init)  # house integration data

    def integrate(self):
        """ Determine when to stop """
        t_start = process_time()
        # Pilot Sample
        self.data.update_data()
        self.sigma_up = self.inflate * self.data.sighat
        self.alpha_mu = 1 - (1 - self.alpha) / (1 - self.alpha_sigma)
        if self.rel_tol == 0:
            toloversig = self.abs_tol / self.sigma_up
            # absolute error tolerance over sigma
            n, self.err_bar = \
                self._nchebe(toloversig, self.alpha_mu, self.kurtmax, self.n_max, self.sigma_up)
            self.data.n[:] = n
            if self.data.n_total + self.data.n > self.n_max:
                # cannot generate this many new samples
                n_low = int(self.n_max-self.data.n_total)
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples would exceeds n_max = %d.
                Will instead generate %d samples to meet n_max total samples. 
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total), int(self.data.n), int(self.n_max), n_low)
                warnings.warn(warning_s, MaxSamplesWarning)
                self.data.n[:] = n_low
        else:
            raise NotYetImplemented("Not implemented for rel_tol != 0")
        # Final Sample
        self.data.update_data()
        self.data.confid_int = self.data.solution + self.err_bar * array([-1, 1])
        self.data.time_integrate = process_time() - t_start
        return self.data.solution, self.data

    def _nchebe(self, toloversig, alpha, kurtmax, n_budget, sigma_0_up):
        """
        This method uses Chebyshev and Berry-Esseen Inequality to calculate the
        sample size needed
        """
        ncheb = ceil(1 / (alpha * toloversig**2))  # sample size by Chebyshev's Inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429  # three constants in Berry-Esseen inequality
        M3upper = kurtmax**(3 / 4)
        # the upper bound on the third moment by Jensen's inequality

        def BEfun2(logsqrtn): return norm.cdf(-exp(logsqrtn) * toloversig) \
            + exp(-logsqrtn) * minimum(A1 * (M3upper + A2),
            A * M3upper / (1 + (exp(logsqrtn) * toloversig)**3)) \
            - alpha / 2
        # Berry-Esseen function, whose solution is the sample size needed
        logsqrtnCLT = log(norm.ppf(1 - alpha / 2) / toloversig)  # sample size by CLT
        nbe = ceil(exp(2 * fsolve(BEfun2, logsqrtnCLT)))
        # calculate Berry-Esseen n by fsolve function (scipy)
        ncb = min(min(ncheb, nbe), n_budget)  # take the min of two sample sizes
        logsqrtn = log(sqrt(ncb))

        def BEfun3(toloversig): return norm.cdf(-exp(logsqrtn) * toloversig) \
            + exp(-logsqrtn) * min(A1 * (M3upper + A2),
            A * M3upper / (1 + (exp(logsqrtn) * toloversig)**3)) \
            - alpha / 2
        err = fsolve(BEfun3, toloversig) * sigma_0_up
        return ncb, err


# To be used when rel_tol != 0
'''
    def _ncbinv(self, alpha1, kurtmax, n1=2):
        """
        Calculate the reliable upper bound on error when given
        Chebyshev and Berry-Esseen inequality and sample size n
        """
        NCheb_inv = 1/sqrt(n1*alpha1)
        # use Chebyshev inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429 # three constants in Berry-Esseen inequality
        M3upper=kurtmax**(3/4)
        # using Jensen's inequality to bound the third moment
        BEfun = lambda logsqrtb: \
            norm.cdf(n1*logsqrtb) + min(A1*(M3upper+A2), \
            A*M3upper/(1+(sqrt(n1)*logsqrtb)**3))/sqrt(n1) \
            - alpha1/2
        # Berry-Esseen inequality
        logsqrtb_clt = log(sqrt(norm.ppf(1-alpha1/2)/sqrt(n1)))
        # use CLT to get tolerance
        NBE_inv = exp(2*fsolve(BEfun,logsqrtb_clt))
        # use fsolve to get Berry-Esseen tolerance
        eps = min(NCheb_inv,NBE_inv)
        # take the min of Chebyshev and Berry Esseen tolerance
        return eps
'''
