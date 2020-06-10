from ._integrand import Integrand
from ..true_measure import Gaussian
from ..util import ParameterError
from numpy import sqrt, exp, log, zeros, maximum, minimum, tile, random, eye, log2
from scipy.stats import norm


class MLCallOptions(Integrand):
    """
    Various call options from finance using Milstein discretization with 2^l timesteps on level l

    Reference:
        M.B. Giles. `Improved multilevel Monte Carlo convergence using the Milstein scheme'.
        343-358, in Monte Carlo and Quasi-Monte Carlo Methods 2006, Springer, 2008.
        http://people.maths.ox.ac.uk/~gilesm/files/mcqmc06.pdf.
    """

    parameters = ['option', 'sigma', 'k', 'r', 't', 'b']

    def __init__(self, measure, option='european', volatility=.2,
        start_strike_price=100, interest_rate=.05, t_final=1):
        """
        Args:
            measure (TrueMeasure): A BrownianMotion TrueMeasure object
            option_type (str): type of option in
            volatility (float): sigma, the volatility of the asset
            start_strike_price (float): S(0), the asset value at t=0, and K, the strike price. 
                Assume start_price = strike_price
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
        """
        if not (isinstance(measure,Gaussian) and all(measure.mu==0) and all(measure.covariance==eye(measure.d))):
            raise ParameterError('AsianCall measure must be a BrownianMotion instance')
        options = ['european','asian']
        self.option = option.lower()
        if self.option not in options:
            raise ParameterError('option type must be one of\n\t%s'%str(options)) 
        self.measure = measure
        self.sigma = volatility
        self.k = start_strike_price
        self.r = interest_rate
        self.t = t_final
        self.b = .85*self.k
        self.multilevel = True
        self.g_submodule = getattr(self,'g_'+self.option)
        super().__init__()

    def get_exact_value(self):
        """ Print exact analytic value, based on s0=k """
        d1 = (self.r+.5*self.sigma**2)*self.t / (self.sigma*sqrt(self.t))
        d2 = (self.r-0.5*self.sigma**2)*self.t / (self.sigma*sqrt(self.t))
        if self.option == 'european':
            val = self.k*( norm.cdf(d1) - exp(-self.r*self.t)*norm.cdf(d2) )
        elif self.option == 'asian':
            print('Exact value unknown for asian option')
            val = None
        elif self.option == 'lookback':
            kk = .5*self.sigma**2/r
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
    
    def g_european(self, samples, l, n, d, nf, nc, hf, hc, xf, xc):
        """
        Implementation for European call option. 
        
        Args:
            samples (ndarray): nxd array of samples
            l (int): level
            n (int): number of samples
            d (int): number of dimensions
            nf (int): n fine samples = 2**level
            nc (int): n coarse sampes = nf/2
            hf (int): fine timestep = self.t/nf
            hc (float): coarse timestep = self.t/nc
            xf (ndarray): n vector of fine samples values = self.k
            xc (ndarray): n vector of coarse samples = self.k

        Return:
            tuple: tuple contining:
                - pf (ndarray): payoffs from fine paths
                - pc (ndarray): payoffs from coarse paths
        """
        dwf = samples * sqrt(hf)
        if l == 0:
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

    def g_asian(self, samples, l, n, d, nf, nc, hf, hc, xf, xc):
        """
        Implementation for Asian call option. 
        
        Args:
            samples (ndarray): nxd array of samples
            l (int): level
            n (int): number of samples
            d (int): number of dimensions
            nf (int): n fine samples = 2**level
            nc (int): n coarse sampes = nf/2
            hf (int): fine timestep = self.t/nf
            hc (float): coarse timestep = self.t/nc
            xf (ndarray): n vector of fine samples values = self.k
            xc (ndarray): n vector of coarse samples = self.k  
        
        Return:
            tuple: tuple contining:
                - pf (ndarray): payoffs from fine paths
                - pc (ndarray): payoffs from coarse paths
        """
        af = .5*hf*xf
        ac = .5*hc*xc
        dwf = sqrt(hf) * samples[:,:int(d/2)]
        dif = sqrt(hf/12) * hf * samples[:,int(d/2):]
        if l == 0:
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
    
    def g(self, samples, l):
        """ See abstract method. """
        n,d = samples.shape        
        nf = 2**l # n fine
        nc = nf/2 # n coarse
        hf = self.t/nf # timestep fine
        hc = self.t/nc # timestep coarse
        xf = tile(self.k,int(n))
        xc = xf
        pf,pc = self.g_submodule(samples, l, n, d, nf, nc, hf, hc, xf, xc)
        dp = exp(-self.r*self.t)*(pf-pc)
        pf = exp(-self.r*self.t)*pf
        if l == 0:
            dp = pf
        sums = zeros(6)
        sums[0] = dp.sum()
        sums[1] = (dp**2).sum()
        sums[2] = (dp**3).sum()
        sums[3] = (dp**4).sum()
        sums[4] = pf.sum()
        sums[5] = (pf**2).sum()    
        cost = n*nf # cost defined as number of fine timesteps
        return sums,cost
    
    def dim_at_level(self, l):
        """ See abstract method. """
        if self.option == 'european':
            return 2**l
        elif self.option == 'asian':
            return 2**(l+1)