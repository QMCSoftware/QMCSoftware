""" Definition for MeanMC_g, a concrete implementation of StoppingCriterion """

from time import time
from numpy import array, zeros, tile, min, exp, sqrt, ceil, log
from scipy.stats import norm
from scipy.optimize import fsolve

from qmcpy._util import MaxSamplesWarning
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
        self.data.n_total = zeros(n_integrands)
        self.stage = "sigma"  # use inital runtime
    
    def stop_yet(self):
        raise Exception("mean_mc_g INCOMPLETE. NOT IMPLEMENTED YET")
        
    def stop_yet_INCOMPLETE(self):
        """ Determine when to stop """
        if self.stage == "sigma":
            sig0up = self.inflate*self.data.sighat # upper bound on the standard deviation
            alpha_sig = self.alpha/2 # the uncertainty for variance estimation
            kurtmax = (self.n_init-3)/(self.n_init-1) + \
                        ((alpha_sig*self.n_init)/(1-alpha_sig)) * \
                        (1-1/self.inflate**2)**2
            #    the upper bound on the modified kurtosis
            if self.reltol == 0:
                alphai = 1-(1-self.alpha)/(1-alpha_sig)
                toloversig = self.abs_tol/sig0up
                # absolute error tolerance over sigma
                n, self.bound_error = _nchebe(toloversig,alphai,kurtmax,sig0up)
                self.data.n = tile(n,len(self.data.n))
                self.stage = 'mu'
            else:
                alphai = (self.alpha-alpha_sig)/(2*(1-alpha_sig))
                #   uncertainty to do iteration
                eps1 = _ncbinv(alphai,kurtmax)
                #   tolerance for initial estimation
                self.bound_err = array([sig0up*eps1])
                #   the width of initial confidence interval for the mean
                errtype = 'max' # see the function tolfun for more info
                theta  = 0 # relative error case
                temp_tol_1 = tolfun(self.abs_tol, self.rel_tol, theta, self.data.solution-self.bound_err, errtype)
                temp_tol_2 = tolfun(self.abs_tol, self.rel_tol, theta, self.data.solution+self.bound_err, errtype)
                deltaplus = (temp_tol_1+temp_tol_2)/2
                #   a combination of tolfun, which used to decide stopping time
                if deltaplus >= self.bound_err: # stopping criterion met
                    self.stage = 'mu'
                else:
                    self.bound_err = min(self.bound_err/2,\
                        max(self.abs_tol, 0.95*self.rel_tol*abs(self.data.solution)))
                    toloversig = self.bound_err/sig0u # next tolerance over sigma
                    alphai = (self.alpha-alpha_sig)/(1-alpha_sig)*2**(-i)
                    # update the next uncertainty
                    self.data.n,_ignore = _nchebe(toloversig,alphai,kurtmax,self.n_max,sig0up)
                    #   get the next sample size needed
        elif self.stage == "mu":
            deltaminus= (temp_tol_1-temp_tol_2)/2
                    #   the other combination of tolfun, which adjusts the hmu a bit
            self.solution = self.solution+deltaminus   

    def _nchebe(self, toloversig, alpha, kurtmax, nbudget, sig0up):
        """
        This method uses Chebyshev and Berry-Esseen Inequality to calculate the
        sample size needed
        """
        ncheb = ceil(1/(toloversig^2*alpha)) # sample size by Chebyshev's Inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429 # three constants in Berry-Esseen inequality
        M3upper = kurtmax**(3/4)
        # the upper bound on the third moment by Jensen's inequality
        BEfun2 = lambda logsqrtn: \
            norm.cdf(-exp(logsqrtn)*toloversig) \
            + exp(-logsqrtn) * min(A1*(M3upper+A2), \
            A*M3upper/(1+(exp(logsqrtn)*toloversig)**3)) \
            - alpha/2
        # Berry-Esseen function, whose solution is the sample size needed
        logsqrtnCLT = log(norm.ppf(1-alpha/2)/toloversig) # sample size by CLT
        nbe = ceil(exp(2*fsolve(BEfun2,logsqrtnCLT)))
        # calculate Berry-Esseen n by fsolve function (scipy)
        ncb = min(min(ncheb,nbe),nbudget) # take the min of two sample sizes
        logsqrtn = log(sqrt(ncb))
        BEfun3 = lambda toloversig: \
            norm.cdf(-exp(logsqrtn)*toloversig) \
            + exp(-logsqrtn)*min(A1*(M3upper+A2), \
            A*M3upper/(1+(exp(logsqrtn)*toloversig)**3)) \
            -alpha/2
        err = fsolve(BEfun3,toloversig) * sig0up
        return ncb, err

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


''' MAYBE PUT IN SEPERATE FILE??? '''
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
