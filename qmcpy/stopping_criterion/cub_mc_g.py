from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MeanVarData
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution._discrete_distribution import IID
from ..integrand import Keister, CustomFun
from ..true_measure import Gaussian, Uniform
from ..discrete_distribution import IIDStdUniform
from ..util import _tol_fun, MaxSamplesWarning
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
from time import time
import warnings


class CubMCG(StoppingCriterion):
    r"""
    Stopping criterion with guaranteed accuracy.

    >>> k = Keister(IIDStdUniform(2,seed=7))
    >>> sc = CubMCG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> data
    MeanVarData (AccumulateData Object)
        solution        1.807
        error_bound     0.050
        n_total         15256
        n               14232
        levels          1
        time_integrate  ...
    CubMCG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           10000000000
        inflate         1.200
        alpha           0.010
    Keister (Integrand Object)
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    IIDStdUniform (DiscreteDistribution Object)
        d               2^(1)
        entropy         7
        spawn_key       ()
    >>> dd = IIDStdUniform(1,seed=7)
    >>> k = Keister(dd)
    >>> cv1 = CustomFun(Uniform(dd),lambda x: np.sin(np.pi*x).sum(1))
    >>> cv1mean = 2/np.pi
    >>> cv2 = CustomFun(Uniform(dd),lambda x: (-3*(x-.5)**2+1).sum(1))
    >>> cv2mean = 3/4
    >>> sc1 = CubMCG(k,abs_tol=.05,control_variates=[cv1,cv2],control_variate_means=[cv1mean,cv2mean])
    >>> sol,data = sc1.integrate()
    >>> print("%.4f"%sol)
    1.3841

    Original Implementation:

        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/meanMC_g.m

    References:

        [1] Fred J. Hickernell, Lan Jiang, Yuewei Liu, and Art B. Owen, "Guaranteed
        conservative fixed width confidence intervals via Monte Carlo
        sampling," Monte Carlo and Quasi-Monte Carlo Methods 2012 (J. Dick, F.
        Y. Kuo, G. W. Peters, and I. H. Sloan, eds.), pp. 105-128,
        Springer-Verlag, Berlin, 2014. DOI: 10.1007/978-3-642-41095-6_5

        [2] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou,
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019.
        Available from http://gailgithub.github.io/GAIL_Dev/

    Guarantee:
        This algorithm attempts to calculate the mean, mu, of a random variable
        to a prescribed error tolerance, _tol_fun:= max(abstol,reltol*|mu|), with
        guaranteed confidence level 1-alpha. If the algorithm terminates without
        showing any warning messages and provides an answer tmu, then the follow
        inequality would be satisfied: $\P(| mu - tmu | \\le tol\_fun) \\ge 1 - \\alpha$.
    """

    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=1024., n_max=1e10,
                 inflate=1.2, alpha=0.01, control_variates=[], control_variate_means=[]):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate: inflation factor when estimating variance
            alpha: significance level for confidence interval
            abs_tol: absolute error tolerance
            rel_tol: relative error tolerance
            n_init: initial number of samples
            n_max: maximum number of samples
            control_variates (list): list of integrand objects to be used as control variates.
                Control variates are currently only compatible with single level problems. 
                The same discrete distribution instance must be used for the integrand and each of the control variates. 
            control_variate_means (list): list of means for each control variate
        """
        self.parameters = ['abs_tol','rel_tol','n_init','n_max','inflate','alpha']
        # Set Attributes
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.alpha = float(alpha)
        self.inflate = float(inflate)
        self.alpha_sigma = float(self.alpha) / 2.  # the uncertainty for variance estimation
        self.kurtmax = (n_init - 3) / (n_init - 1) + \
            (self.alpha_sigma * n_init) / (1 - self.alpha_sigma) * \
            (1 - 1. / self.inflate**2)**2
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.cv = control_variates
        self.cv_mu = control_variate_means
        # Verify Compliant Construction
        allowed_levels = ['single']
        allowed_distribs = [IID]
        allow_vectorized_integrals = False
        super(CubMCG,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

    def integrate(self):
        """ See abstract method. """
        # Construct AccumulateData Object to House Integration data
        self.data = MeanVarData(self, self.integrand, self.true_measure, self.discrete_distrib, 
            self.n_init, self.cv, self.cv_mu)  # house integration data
        t_start = time()
        # Pilot Sample
        self.data.update_data()
        self.sigma_up = self.inflate * self.data.sighat
        if self.rel_tol == 0:
            self.alpha_mu = 1 - (1 - self.alpha) / (1 - self.alpha_sigma)
            toloversig = self.abs_tol / self.sigma_up
            # absolute error tolerance over sigma
            n, self.data.error_bound = \
                self._nchebe(toloversig, self.alpha_mu, self.kurtmax, self.n_max, self.sigma_up)
            self.data.n[:] = n
            if self.data.n_total + self.data.n > self.n_max:
                # cannot generate this many new samples
                n_low = int(self.n_max-self.data.n_total)
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceeds n_max = %d.
                Will instead generate %d samples to meet n_max total samples. 
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total), int(self.data.n), int(self.n_max), n_low)
                warnings.warn(warning_s, MaxSamplesWarning)
                self.data.n[:] = n_low
            self.data.update_data()
        else: # self.rel_tol > 0
            alphai = (self.alpha-self.alpha_sigma)/(2*(1-self.alpha_sigma)) # uncertainty to do iteration
            eps1 = self._ncbinv(1e4,alphai,self.kurtmax)
            self.data.error_bound = self.sigma_up*eps1
            tau = 1. # step of the iteration
            self.data.n[:] = 1e4 # default initial sample size
            while True:
                if self.data.n_total + self.data.n > self.n_max:
                    # cannot generate this many new samples
                    n_low = int(self.n_max-self.data.n_total)
                    warning_s = """
                    Already generated %d samples.
                    Trying to generate %d new samples would exceeds n_max = %d.
                    Will instead generate %d samples to meet n_max total samples. 
                    Note that error tolerances may no longer be satisfied""" \
                    % (int(self.data.n_total), int(self.data.n), int(self.n_max), n_low)
                    warnings.warn(warning_s, MaxSamplesWarning)
                    self.data.n[:] = n_low
                    self.data.update_data()
                    break
                self.data.update_data()
                lb_tol = _tol_fun(self.abs_tol, self.rel_tol, 0., self.data.solution-self.data.error_bound, 'max')
                ub_tol = _tol_fun(self.abs_tol, self.rel_tol, 0., self.data.solution+self.data.error_bound, 'max')
                delta_plus = (lb_tol + ub_tol) / 2.
                if delta_plus >= self.data.error_bound:
                    # stopping criterion met
                    delta_minus = (lb_tol - ub_tol) / 2.
                    self.data.solution += delta_minus # adjust solution a bit
                    break
                else:
                    candidate_tol = np.maximum(self.abs_tol,.95*self.rel_tol*abs(self.data.solution))
                    self.data.error_bound = np.minimum(self.data.error_bound/2.,candidate_tol)
                    tau += 1
                # update next uncertainty
                toloversig = self.data.error_bound / self.sigma_up
                alphai = 2**tau * (self.alpha - self.alpha_sigma) / (1 - self.alpha_sigma)
                n,_ = self._nchebe(toloversig, alphai, self.kurtmax, self.n_max, self.sigma_up)
                self.data.n[:] = n
        # set confidence interval
        self.data.confid_int = self.data.solution + self.data.error_bound * np.array([-1, 1])
        self.data.time_integrate = time() - t_start
        return self.data.solution, self.data

    def _nchebe(self, toloversig, alpha, kurtmax, n_budget, sigma_0_up):
        ncheb = np.ceil(1 / (alpha * toloversig**2))  # sample size by Chebyshev's Inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429  # three constants in Berry-Esseen inequality
        M3upper = kurtmax**(3. / 4)
        # the upper bound on the third moment by Jensen's inequality
        BEfun2 = lambda logsqrtn: \
            (norm.cdf(-np.exp(logsqrtn) * toloversig)
            + np.exp(-logsqrtn) * np.minimum(A1 * (M3upper + A2),
            A * M3upper / (1 + (np.exp(logsqrtn) * toloversig)**3))
            - alpha / 2.)
        # Berry-Esseen function, whose solution is the sample size needed
        logsqrtnCLT = np.log(norm.ppf(1 - alpha / 2) / toloversig)  # sample size by CLT
        nbe = np.ceil(np.exp(2 * fsolve(BEfun2, logsqrtnCLT)))
        # calculate Berry-Esseen n by fsolve function (scipy)
        ncb = np.minimum(np.minimum(ncheb, nbe), n_budget)  # take the min of two sample sizes
        logsqrtn = np.log(np.sqrt(ncb))
        BEfun3 = lambda toloversig: \
            (norm.cdf(-np.exp(logsqrtn) * toloversig)
            + np.exp(-logsqrtn) * np.minimum(A1 * (M3upper + A2),
            A * M3upper / (1 + (np.exp(logsqrtn) * toloversig)**3))
            - alpha / 2.)
        err = fsolve(BEfun3, toloversig) * sigma_0_up
        return ncb, err

    def _ncbinv(self, n1, alpha1, kurtmax):
        NCheb_inv = 1/np.sqrt(n1*alpha1)
        # use Chebyshev inequality
        A = 18.1139
        A1 = 0.3328
        A2 = 0.429 # three constants in Berry-Esseen inequality
        M3upper = kurtmax**(3./4)
        # using Jensen's inequality to bound the third moment
        BEfun = lambda logsqrtb: \
            (norm.cdf(n1*logsqrtb) + np.minimum(A1*(M3upper+A2),
            A*M3upper/(1+(np.sqrt(n1)*logsqrtb)**3))/np.sqrt(n1)
            - alpha1/2)
        # Berry-Esseen inequality
        logsqrtb_clt = np.log(np.sqrt(norm.ppf(1-alpha1/2)/np.sqrt(n1)))
        # use CLT to get tolerance
        NBE_inv = np.exp(2*fsolve(BEfun,logsqrtb_clt))
        # use fsolve to get Berry-Esseen tolerance
        eps = np.minimum(NCheb_inv,NBE_inv)
        # take the min of Chebyshev and Berry Esseen tolerance
        return eps

    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method.

        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not.
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not.
        """
        if abs_tol != None: self.abs_tol = abs_tol
        if rel_tol != None: self.rel_tol = rel_tol
