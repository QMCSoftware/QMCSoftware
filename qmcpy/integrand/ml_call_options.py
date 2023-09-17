from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian
from ..util import ParameterError
from numpy import *
from scipy.stats import norm


class MLCallOptions(Integrand):
    """
    Various call options from finance using Milstein discretization with $2^l$ timesteps on level $l$.

    >>> mlco_original = MLCallOptions(DigitalNetB2(seed=7))
    >>> mlco_original
    MLCallOptions (Integrand Object)
        option          european
        sigma           0.200
        k               100
        r               0.050
        t               1
        b               85
        level           0
    >>> mlco_ml_dims = mlco_original.spawn(levels=arange(4))
    >>> yml = 0
    >>> for mlco in mlco_ml_dims:
    ...     x = mlco.discrete_distrib.gen_samples(2**10)
    ...     yml += mlco.f(x).mean()
    >>> yml
    10.393...

    References:

    [1] M.B. Giles. Improved multilevel Monte Carlo convergence using the Milstein scheme.
    343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008.
    http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf.
    """

    def __init__(self, sampler, option='european', volatility=.2,
        start_strike_price=100., interest_rate=.05, t_final=1., _level=0):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            option_type (str): type of option in ["European","Asian"]
            volatility (float): sigma, the volatility of the asset
            start_strike_price (float): S(0), the asset value at t=0, and K, the strike price. \
                Assume start_price = strike_price
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
            _level (int): for internal use only, users should not set this parameter. 

        """
        self.parameters = ['option', 'sigma', 'k', 'r', 't', 'b', 'level']
        self.sampler = sampler
        self.true_measure = Gaussian(self.sampler, mean=0, covariance=1)
        self.discrete_distrib = self.true_measure.discrete_distrib
        options = ['european','asian']
        self.option = option.lower()
        if self.option not in options:
            raise ParameterError('option type must be one of\n\t%s'%str(options))
        #if self.discrete_distrib.low_discrepancy and self.option=='asian':
        #    raise ParameterError('MLCallOptions does not support LD sequence for Asian Option')
        self.sigma = volatility
        self.k = start_strike_price
        self.r = interest_rate
        self.t = t_final
        self.b = .85*self.k
        self.leveltype = 'adaptive-multi'
        self.g_submodule = getattr(self,'_g_'+self.option)
        self.level = _level
        self.max_level = inf
        super(MLCallOptions,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
        #if self.discrete_distrib.low_discrepancy and self.option=='asian':
        #    raise ParameterError('MLCallOptions does not support LD sequence for Asian Option')

    def get_exact_value(self):
        """ Print exact analytic value, based on s0=k. """
        d1 = (self.r+.5*self.sigma**2)*self.t / (self.sigma*sqrt(self.t))
        d2 = (self.r-0.5*self.sigma**2)*self.t / (self.sigma*sqrt(self.t))
        if self.option == 'european':
            val = self.k*( norm.cdf(d1) - exp(-self.r*self.t)*norm.cdf(d2) )
        elif self.option == 'asian':
            print('Exact value unknown for asian option')
            val = None
        elif self.option == 'lookback':
            kk = .5*self.sigma**2/self.r
            val = self.k*( norm.cdf(d1) - norm.cdf(-d1)*kk -
                      exp(-self.r*self.t)*(norm.cdf(d2) - norm.cdf(d2)*kk) )
        elif self.option == 'digital':
            val = self.k*exp(-self.r*self.t)*norm.cdf(d2)
        elif self.option == 'barrier':
            kk = .5*self.sigma**2/self.r
            d3 = (2*log(self.b/self.k) + (self.r+.5*self.sigma**2)*self.t) / (self.sigma*sqrt(self.t))
            d4 = (2*log(self.b/self.k) + (self.r-.5*self.sigma**2)*self.t) / (self.sigma*sqrt(self.t))
            val = self.k*( norm.cdf(d1) - exp(-self.r*self.t)*norm.cdf(d2) -
                     (self.k/self.b)**(1-1/kk)*( (self.b/self.k)**2*norm.cdf(d3) -
                     exp(-self.r*self.t)*norm.cdf(d4) ) )
        return val

    def _g_european(self, t, n, d, nf, nc, hf, hc, xf, xc):
        """
        Implementation for European call option.

        Args:
            t (ndarray): nxd array of samples
            n (int): number of samples
            d (int): number of dimensions
            nf (int): n fine samples = 2**level
            nc (int): n coarse samples = nf/2
            hf (int): fine timestep = self.t/nf
            hc (float): coarse timestep = self.t/nc
            xf (ndarray): n vector of fine samples values = self.k
            xc (ndarray): n vector of coarse samples = self.k

        Return:
            tuple: \
                First, an ndarray of payoffs from fine paths. \
                Second, an ndarray of payoffs from coarse paths.
        """
        dwf = t * sqrt(hf)
        if self.level == 0:
            dwf = dwf.squeeze()
            xf = xf + self.r*xf*hf + self.sigma*xf*dwf + .5*self.sigma**2*xf*(dwf**2-hf)
        else:
            for j in range(int(nf)):
                xf = xf + self.r*xf*hf + self.sigma*xf*dwf[:,j] + .5*self.sigma**2*xf*(dwf[:,j]**2-hf)
                if (j%2)==1:
                    dwc = dwf[:,j-1] + dwf[:,j]
                    ddw = dwf[:,j-1] - dwf[:,j]
                    xc = xc + self.r*xc*hc + self.sigma*xc*dwc + .5*self.sigma**2*xc*(dwc**2-hc)
        pf = maximum(0,xf-self.k)
        pc = maximum(0,xc-self.k)
        return pf,pc

    def _g_asian(self, t, n, d, nf, nc, hf, hc, xf, xc):
        """
        Implementation for Asian call option.

        Args:
            t (ndarray): nxd array of samples
            n (int): number of samples
            d (int): number of dimensions
            nf (int): n fine samples = 2**level
            nc (int): n coarse samples = nf/2
            hf (int): fine timestep = self.t/nf
            hc (float): coarse timestep = self.t/nc
            xf (ndarray): n vector of fine samples values = self.k
            xc (ndarray): n vector of coarse samples = self.k

        Return:
            tuple: \
                First, an ndarray of payoffs from fine paths. \
                Second, an ndarray of payoffs from coarse paths.
        """
        af = .5*hf*xf
        ac = .5*hc*xc
        dwf = sqrt(hf) * t[:,:int(d/2)]
        dif = sqrt(hf/12) * hf * t[:,int(d/2):]
        if self.level == 0:
            dwf = dwf.squeeze()
            dif = dif.squeeze()
            xf0 = xf
            xf = xf + self.r*xf*hf + self.sigma*xf*dwf + .5*self.sigma**2*xf*(dwf**2-hf)
            vf = self.sigma*xf0
            af = af + .5*hf*xf + vf*dif
        else:
            for j in range(int(nf)):
                xf0 = xf
                xf = xf + self.r*xf*hf + self.sigma*xf*dwf[:,j] + .5*self.sigma**2*xf*(dwf[:,j]**2-hf)
                vf = self.sigma*xf0
                af = af + hf*xf + vf*dif[:,j]
                if (j%2)==1:
                    dwc = dwf[:,j-1] + dwf[:,j]
                    ddw = dwf[:,j-1] - dwf[:,j]
                    xc0 = xc
                    xc = xc + self.r*xc*hc + self.sigma*xc*dwc + .5*self.sigma**2*xc*(dwc**2-hc)
                    vc = self.sigma*xc0
                    dif_cs = dif[:,j-1] + dif[:,j]
                    ac = ac + hc*xc + vc*(dif_cs + .25*hc*ddw)
            af = af - 0.5*hf*xf
            ac = ac - 0.5*hc*xc
        pf = maximum(0,af-self.k)
        pc = maximum(0,ac-self.k)
        return pf,pc

    def g(self, t):
        """
        Args:
            t (ndarray): Gaussian(0,1^2) samples

        Returns:
            tuple: \
                First, an ndarray of length 6 vector of summary statistic sums. \
                Second, a float of cost on this level.
        """
        n,d = t.shape
        nf = 2**self.level # n fine
        nc = float(nf)/2 # n coarse
        hf = self.t/nf # timestep fine
        hc = self.t/nc # timestep coarse
        xf = tile(self.k,int(n))
        xc = xf
        pf,pc = self.g_submodule(t, n, d, nf, nc, hf, hc, xf, xc)
        dp = exp(-self.r*self.t)*(pf-pc)
        pf = exp(-self.r*self.t)*pf
        if self.level == 0:
            dp = pf
        sums = zeros(6)
        sums[0] = dp.sum()
        sums[1] = (dp**2).sum()
        sums[2] = (dp**3).sum()
        sums[3] = (dp**4).sum()
        sums[4] = pf.sum()
        sums[5] = (pf**2).sum()
        cost = n*nf # cost defined as number of fine timesteps
        self.cost = cost
        self.sums = sums
        return dp

    def _dimension_at_level(self, level):
        """ See abstract method. """
        if self.option == 'european':
            return 2**level
        elif self.option == 'asian':
            return 2**(level+1)
    
    def _spawn(self, level, sampler):
        return MLCallOptions(
            sampler = sampler,
            option = self.option,
            volatility = self.sigma,
            start_strike_price = self.k,
            interest_rate = self.r,
            t_final = self.t,
            _level = level)
