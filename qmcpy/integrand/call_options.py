"""
Various call options from finance 
using Milstein discretization with 2^l timesteps on level l
"""

from ._integrand import Integrand
from ..true_measure import Gaussian
from ..util import ParameterError
from numpy import sqrt, exp, log, zeros, maximum, minimum, tile, random, eye, log2
from scipy.stats import norm


class CallOptions(Integrand):
    """ Specify ad generate payoff values from various call options """

    parameters = []

    def __init__(self, measure, option='european', volatility=.2, start_strike_price=100, 
        interest_rate=.05, t_final=1):
        """
        Initialize call option object

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
        options = ['european']#,'asian','lookback','digital','barrier']
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
        super().__init__()

    def get_exact_value(self):
        """Print exact analytic value, based on s0=k"""
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
    
    def g(self, dwf, l):
        """
        Original integrand on level l
        Args:
            dwf (ndarray): nxd ndarray for d=2**l as specified by dim_at_level method
            l (int): level
        Return:
            sums (ndarray): length 6 vector of sums such that    
                sums(1) = sum(Pf-Pc)
                sums(2) = sum((Pf-Pc).^2)
                sums(3) = sum((Pf-Pc).^3)
                sums(4) = sum((Pf-Pc).^4)
                sums(5) = sum(Pf)
                sums(6) = sum(Pf.^2)
            cost (float): user-defined computational cost
        """
        n = dwf.shape[0]
        nf = 2**l # n fine
        nc = nf/2 # n coarse
        hf = self.t/nf # timestep fine
        hc = self.t/nc # timestep coarse
        xf = tile(self.k,int(n))
        xc = xf
        dwf = sqrt(hf)*dwf
        ##af = .5*hf*xf
        ##ac = .5*hc*xc
        ##mf = xf
        ##mc = xc
        ##bf = 1
        ##bc = 1
        if l == 0:
            dwf = dwf.squeeze()
            ##xf0 = xf
            xf = xf + self.r*xf*hf + self.sigma*xf*dwf + .5*self.sigma**2*xf*(dwf**2-hf)
            ##lf = log(random.rand(int(n)))
            ##dif = sqrt(hf/12)*hf*random.randn(int(n))
            ##vf = self.sigma*xf0
            ##af = af + .5*hf*xf + vf*dif
            ##mf = minimum(mf,.5*(xf0+xf-sqrt((xf-xf0)**2-2*hf*vf**2*lf)))
            ##bf = bf*(1-exp(-2*maximum(0,(xf0-self.b)*(xf-self.b)/(hf*vf**2))))
        else:
            for j in range(int(nf)):
                ##lf = log(random.rand(2,int(n)))
                ##dif = sqrt(hf/12)*hf*random.randn(2,int(n))
                ##xf0 = xf
                xf = xf + self.r*xf*hf + self.sigma*xf*dwf[:,j] + .5*self.sigma**2*xf*(dwf[:,j]**2-hf)
                ##vf = self.sigma*xf0
                ##af = af + hf*xf + vf*dif[m,:]
                ##mf = minimum(mf,.5*(xf0+xf-sqrt((xf-xf0)**2-2*hf*vf**2*lf[m,:])))
                ##bf = bf*(1-exp(-2*maximum(0,(xf0-self.b)*(xf-self.b)/(hf*vf**2))))
                if (j%2)==1:
                    dwc = dwf[:,j-1] + dwf[:,j]
                    ddw = dwf[:,j-1] - dwf[:,j]
                    ##xc0 = xc
                    xc = xc + self.r*xc*hc + self.sigma*xc*dwc + .5*self.sigma**2*xc*(dwc**2-hc)
                    ##vc = self.sigma*xc0
                    ##ac = ac + hc*xc + vc*(dif.sum(0) + .25*hc*ddw)
                    ##xc1 = .5*(xc0 + xc + vc*ddw)
                    ##mc = minimum(mc, .5*(xc0+xc1-sqrt((xc1- xc0)**2-2*hf*vc**2*lf[0,:])))
                    ##mc = minimum(mc, .5*(xc1+xc -sqrt((xc - xc1)**2-2*hf*vc**2*lf[1,:])))
                    ##bc = bc *(1-exp(-2*maximum(0,(xc0-self.b)*(xc1- self.b)/(hf*vc**2))))
                    ##bc = bc *(1-exp(-2*maximum(0,(xc1-self.b)*(xc - self.b)/(hf*vc**2))))
                ##af = af - 0.5*hf*xf
                ##ac = ac - 0.5*hc*xc
        if self.option == 'european':
            pf = maximum(0,xf-self.k)
            pc = maximum(0,xc-self.k)
        elif self.option == 'asian':
            pf = maximum(0,af-self.k)
            pc = maximum(0,ac-self.k)
        elif self.option == 'lookback':
            pf = xf - mf
            pc = xc - mc
        elif self.option == 'digital':
            if l == 0:
                pf = self.k*norm.cdf((xf0+self.r*xf0*hf-self.k)/(self.sigma*xf0*sqrt(hf)))
                pc = pf
            else:
                pf = self.k*norm.cdf((xf0+self.r*xf0*hf-self.k)/(self.sigma*xf0*sqrt(hf)))
                pc = self.k*norm.cdf((xc0+self.r*xc0*hc+self.sigma*xc0*dwf[0,:]-self.k)/(self.sigma*xc0*sqrt(hf)))
        elif self.option == 'barrier':
            pf = bf*maximum(0,xf-self.k)
            pc = bc*maximum(0,xc-self.k)
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
        """ 
        Get dimension of the SDE at level l
            l (int): level
        """
        return 2**l