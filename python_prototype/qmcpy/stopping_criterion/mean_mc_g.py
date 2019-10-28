""" Definition for MeanMC_g, a concrete implementation of StoppingCriterion """

from time import time
from numpy import array, zeros, tile, minimum, exp, sqrt, ceil, log
from scipy.stats import norm
from scipy.optimize import fsolve

from . import StoppingCriterion
from ..accum_data import MeanVarData

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

    def __init__(self, discrete_distrib, true_measure,
                 inflate=1.2, alpha=0.01,
                 abs_tol=1e-2, rel_tol=0,
                 n_init=1024, n_max=1e8):
        """
        Args:
            discrete_distrib
            true_measure: an instance of DiscreteDistribution
            inflate: inflation factor when estimating variance
            alpha: significance level for confidence interval
            abs_tol: absolute error tolerance
            rel_tol: relative error tolerance
            n_init: initial number of samples
            n_max: maximum number of samples
        """
        allowed_distribs = ["IIDStdUniform",
                            "IIDStdGaussian"]  # supported distributions
        super().__init__(discrete_distrib, allowed_distribs, abs_tol, rel_tol,
                         n_init, n_max)
        self.inflate = inflate  # inflation factor
        self.alpha = alpha  # uncertainty level
        # Construct Data Object
        n_integrands = len(true_measure)
        self.data = MeanVarData(n_integrands)  # house integration data
        self.data.n = tile(self.n_init,n_integrands)  # next n for each integrand
        self.alpha_sigma = self.alpha/2 # the uncertainty for variance estimation
        self.kurtmax = (self.n_init-3)/(self.n_init-1) + \
                    (self.alpha_sigma*self.n_init)/(1-self.alpha_sigma) * \
                    (1-1/self.inflate**2)**2
        self.stage = "sigma"
    
    def stop_yet(self):
        """ Determine when to stop """
        if self.stage == "sigma":
            self.sigma_up = self.inflate*sqrt((self.data.sighat**2).sum()) # CORRECT? 
            self.alpha_mu = 1-(1-self.alpha)/(1-self.alpha_sigma)
            if self.rel_tol == 0:
                toloversig = self.abs_tol/self.sigma_up
                # absolute error tolerance over sigma
                n, self.err_bar = \
                    self._nchebe(toloversig,self.alpha_mu,self.kurtmax,self.n_max,self.sigma_up)
                self.data.n = tile(n,len(self.data.n))
                self.stage = 'mu'
            else:
                raise Exception("Not yet implemented for rel_tol != 0")
        elif self.stage == "mu":
            self.data.confid_int = self.data.solution + self.err_bar * array([-1, 1])
            self.stage = "done"  # finished with computation

    def _nchebe(self, toloversig, alpha, kurtmax, nbudget, sig0up):
        """
        This method uses Chebyshev and Berry-Esseen Inequality to calculate the
        sample size needed
        """
        ncheb = ceil(1/(alpha*toloversig**2)) # sample size by Chebyshev's Inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429 # three constants in Berry-Esseen inequality
        M3upper = kurtmax**(3/4)
        # the upper bound on the third moment by Jensen's inequality
        BEfun2 = lambda logsqrtn: norm.cdf(-exp(logsqrtn)*toloversig) \
            + exp(-logsqrtn) * minimum(A1*(M3upper+A2), \
            A*M3upper/(1+(exp(logsqrtn)*toloversig)**3)) \
            - alpha/2
        # Berry-Esseen function, whose solution is the sample size needed
        logsqrtnCLT = log(norm.ppf(1-alpha/2)/toloversig) # sample size by CLT
        nbe = ceil(exp(2*fsolve(BEfun2,logsqrtnCLT)))
        # calculate Berry-Esseen n by fsolve function (scipy)
        ncb = min(min(ncheb,nbe),nbudget) # take the min of two sample sizes
        logsqrtn = log(sqrt(ncb))
        BEfun3 = lambda toloversig: norm.cdf(-exp(logsqrtn)*toloversig) \
            + exp(-logsqrtn)*min(A1*(M3upper+A2), \
            A*M3upper/(1+(exp(logsqrtn)*toloversig)**3)) \
            -alpha/2
        err = fsolve(BEfun3,toloversig) * sig0up
        return ncb, err




# Will be used when rel_tol != 0
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


""" MAYBE PUT IN SEPERATE FILE??? """
from numpy import max,abs
def tolfun(abstol, reltol, theta, mu, toltype):
    """
    Generalized error tolerance function.
    
    Args: 
        abstol (float): absolute error tolertance
        reltol (float): relative error tolerance
        theta (float): parameter in 'theta' case
        mu (loat): true mean
        toltype (str): different options of tolerance function
    """
    if toltype == 'combine': # the linear combination of two tolerances
        # theta=0---relative error tolarance
        # theta=1---absolute error tolerance
        tol  = theta*abstol + (1-theta)*reltol*abs(mu)
    elif toltype == 'max': # the max case
        tol  = max(abstol,reltol*abs(mu))
    return tol
'''