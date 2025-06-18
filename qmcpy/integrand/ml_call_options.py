from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian
from ..util import ParameterError
import numpy as np
from scipy.stats import norm


class MLCallOptions(AbstractIntegrand):
    """
    Various call options from finance using Milstein discretization with $2^l$ timesteps on level $l$.

    Examples:
        >>> seed_seq = np.random.SeedSequence(7) 
        >>> initial_level = 3
        >>> num_levels = 4
        >>> ns = [2**11,2**10,2**9,2**8]
        >>> integrands = [MLCallOptions(DigitalNetB2(dimension=2**l,seed=seed_seq.spawn(1)[0]),option="ASIAN") for l in range(initial_level,initial_level+num_levels)]
        >>> ys = [integrands[l](ns[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("ys[%d].shape = %s"%(l,ys[l].shape))
        ys[0].shape = (2, 2048)
        ys[1].shape = (2, 1024)
        ys[2].shape = (2, 512)
        ys[3].shape = (2, 256)
        >>> ymeans = np.stack([(ys[l][1]-ys[l][0]).mean(-1) for l in range(num_levels)])
        >>> ymeans
        array([5.62008251, 5.62798894, 5.63549268, 5.64344119])
        >>> print("%.4f"%ymeans.sum())
        22.5270

        Multi-level options with independent replications
         
        >>> seed_seq = np.random.SeedSequence(7) 
        >>> initial_level = 3
        >>> num_levels = 4
        >>> ns = [2**7,2**6,2**5,2**4]
        >>> integrands = [MLCallOptions(DigitalNetB2(dimension=2**l,seed=seed_seq.spawn(1)[0],replications=2**4),option="ASIAN") for l in range(initial_level,initial_level+num_levels)]
        >>> ys = [integrands[l](ns[l]) for l in range(num_levels)]
        >>> for l in range(num_levels):
        ...     print("ys[%d].shape = %s"%(l,ys[l].shape))
        ys[0].shape = (2, 16, 128)
        ys[1].shape = (2, 16, 64)
        ys[2].shape = (2, 16, 32)
        ys[3].shape = (2, 16, 16)
        >>> muhats = np.stack([(ys[l][1]-ys[l][0]).mean(-1) for l in range(num_levels)])
        >>> muhats.shape
        (4, 16)
        >>> muhathat = muhats.mean(-1)
        >>> muhathat
        array([5.62820834, 5.65408695, 5.60367937, 5.47880253])
        >>> print("%.4f"%muhathat.sum())
        22.3648
        

    References:

    [1] M.B. Giles. Improved multilevel Monte Carlo convergence using the Milstein scheme.
    343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008.
    http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf.
    """

    def __init__(self, sampler, option='european', volatility=.2,
        start_strike_price=100., interest_rate=.05, t_final=1., _level=0):
        """
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): A 
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
        self.max_level = np.inf
        self.cost = sampler.d
        super(MLCallOptions,self).__init__(dimension_indv=(2,),dimension_comb=(2,),parallel=False)
        #if self.discrete_distrib.low_discrepancy and self.option=='asian':
        #    raise ParameterError('MLCallOptions does not support LD sequence for Asian Option')

    def get_exact_value(self):
        """ Print exact analytic value, based on s0=k. """
        d1 = (self.r+.5*self.sigma**2)*self.t / (self.sigma*np.sqrt(self.t))
        d2 = (self.r-0.5*self.sigma**2)*self.t / (self.sigma*np.sqrt(self.t))
        if self.option == 'european':
            val = self.k*( norm.cdf(d1) - np.exp(-self.r*self.t)*norm.cdf(d2) )
        elif self.option == 'asian':
            print('Exact value unknown for asian option')
            val = None
        elif self.option == 'lookback':
            kk = .5*self.sigma**2/self.r
            val = self.k*( norm.cdf(d1) - norm.cdf(-d1)*kk -
                      np.exp(-self.r*self.t)*(norm.cdf(d2) - norm.cdf(d2)*kk) )
        elif self.option == 'digital':
            val = self.k*np.exp(-self.r*self.t)*norm.cdf(d2)
        elif self.option == 'barrier':
            kk = .5*self.sigma**2/self.r
            d3 = (2*np.log(self.b/self.k) + (self.r+.5*self.sigma**2)*self.t) / (self.sigma*np.sqrt(self.t))
            d4 = (2*np.log(self.b/self.k) + (self.r-.5*self.sigma**2)*self.t) / (self.sigma*np.sqrt(self.t))
            val = self.k*( norm.cdf(d1) - np.exp(-self.r*self.t)*norm.cdf(d2) -
                     (self.k/self.b)**(1-1/kk)*( (self.b/self.k)**2*norm.cdf(d3) -
                     np.exp(-self.r*self.t)*norm.cdf(d4) ) )
        return val

    def _g_european(self, t, n, d, nf, nc, hf, hc, xf, xc):
        """
        Implementation for European call option.

        Args:
            t (np.ndarray): nxd array of samples
            n (int): number of samples
            d (int): number of dimensions
            nf (int): n fine samples = 2**level
            nc (int): n coarse samples = nf/2
            hf (int): fine timestep = self.t/nf
            hc (float): coarse timestep = self.t/nc
            xf (np.ndarray): n vector of fine samples values = self.k
            xc (np.ndarray): n vector of coarse samples = self.k

        Returns:
            tuple: \
                First, an np.ndarray of payoffs from fine paths. \
                Second, an np.ndarray of payoffs from coarse paths.
        """
        dwf = t * np.sqrt(hf)
        if self.level == 0:
            dwf = dwf[...,0]
            xf = xf + self.r*xf*hf + self.sigma*xf*dwf + .5*self.sigma**2*xf*(dwf**2-hf)
        else:
            for j in range(int(nf)):
                xf = xf + self.r*xf*hf + self.sigma*xf*dwf[...,j] + .5*self.sigma**2*xf*(dwf[...,j]**2-hf)
                if (j%2)==1:
                    dwc = dwf[...,j-1] + dwf[...,j]
                    ddw = dwf[...,j-1] - dwf[...,j]
                    xc = xc + self.r*xc*hc + self.sigma*xc*dwc + .5*self.sigma**2*xc*(dwc**2-hc)
        pf = np.maximum(0,xf-self.k)
        pc = np.maximum(0,xc-self.k)
        return pf,pc

    def _g_asian(self, t, n, d, nf, nc, hf, hc, xf, xc):
        """
        Implementation for Asian call option.

        Args:
            t (np.ndarray): nxd array of samples
            n (int): number of samples
            d (int): number of dimensions
            nf (int): n fine samples = 2**level
            nc (int): n coarse samples = nf/2
            hf (int): fine timestep = self.t/nf
            hc (float): coarse timestep = self.t/nc
            xf (np.ndarray): n vector of fine samples values = self.k
            xc (np.ndarray): n vector of coarse samples = self.k

        Returns:
            tuple: \
                First, an np.ndarray of payoffs from fine paths. \
                Second, an np.ndarray of payoffs from coarse paths.
        """
        assert t.shape[-1]>1, "asian option requires at least d=2 timesteps"
        af = .5*hf*xf
        ac = .5*hc*xc
        dwf = np.sqrt(hf) * t[...,:int(d/2)]
        dif = np.sqrt(hf/12) * hf * t[...,int(d/2):]
        if self.level == 0:
            dwf = dwf[...,0]
            dif = dif[...,0]
            xf0 = xf
            xf = xf + self.r*xf*hf + self.sigma*xf*dwf + .5*self.sigma**2*xf*(dwf**2-hf)
            vf = self.sigma*xf0
            af = af + .5*hf*xf + vf*dif
        else:
            for j in range(int(nf)):
                xf0 = xf
                xf = xf + self.r*xf*hf + self.sigma*xf*dwf[...,j] + .5*self.sigma**2*xf*(dwf[...,j]**2-hf)
                vf = self.sigma*xf0
                af = af + hf*xf + vf*dif[...,j]
                if (j%2)==1:
                    dwc = dwf[...,j-1] + dwf[...,j]
                    ddw = dwf[...,j-1] - dwf[...,j]
                    xc0 = xc
                    xc = xc + self.r*xc*hc + self.sigma*xc*dwc + .5*self.sigma**2*xc*(dwc**2-hc)
                    vc = self.sigma*xc0
                    dif_cs = dif[...,j-1] + dif[...,j]
                    ac = ac + hc*xc + vc*(dif_cs + .25*hc*ddw)
            af = af - 0.5*hf*xf
            ac = ac - 0.5*hc*xc
        pf = np.maximum(0,af-self.k)
        pc = np.maximum(0,ac-self.k)
        return pf,pc

    def g(self, t, compute_flags=None):
        """
        Args:
            t (np.ndarray): Gaussian(0,1^2) samples

        Returns:
            tuple: \
                First, an np.ndarray of length 6 vector of summary statistic sums. \
                Second, a float of cost on this level.
        """
        n,d = t.shape[-2:]
        nf = 2**self.level # n fine
        nc = float(nf)/2 # n coarse
        hf = self.t/nf # timestep fine
        hc = self.t/nc # timestep coarse
        xf = np.tile(self.k,t.shape[:-1])
        xc = xf
        pf,pc = self.g_submodule(t, n, d, nf, nc, hf, hc, xf, xc)
        pf = np.exp(-self.r*self.t)*pf
        if self.level == 0:
            pc = np.zeros_like(pf)
        else:
            pc = np.exp(-self.r*self.t)*pc
        return np.stack([pc,pf],axis=0)

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
