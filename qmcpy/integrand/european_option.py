from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
from numpy import *
from scipy.stats import norm 
from ._option import Option


class EuropeanOption(Option):
    """
    European financial option. 

    >>> eo = EuropeanOption(DigitalNetB2(4,seed=7),call_put='put')
    >>> eo
    EuropeanOption (Integrand Object)
        volatility      2^(-1)
        call_put        put
        start_price     30
        strike_price    35
        interest_rate   0
    >>> x = eo.discrete_distrib.gen_samples(2**12)
    >>> y = eo.f(x)
    >>> y.mean()
    9.209...
    >>> eo = EuropeanOption(BrownianMotion(DigitalNetB2(4,seed=7),drift=1),call_put='put')
    >>> x = eo.discrete_distrib.gen_samples(2**12)
    >>> y = eo.f(x)
    >>> y.mean()
    9.162...
    >>> eo.get_exact_value()
    9.211452976234058
    """
                          
    def __init__(self, sampler, volatility=0.5, start_price=30.0, strike_price=35.0,
        interest_rate=0.0, t_final=1, call_put='call'):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
            call_put (str): 'call' or 'put' option
        """
        super(EuropeanOption, self).__init__(sampler, volatility, start_price,
                                          strike_price, interest_rate, t_final,
                                          call_put, multilevel_dims=None, dim_frac=1)

    def g(self, t):
        """ See abstract method. """
        self.s = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.true_measure.time_vec + self.volatility * t)
        for xx,yy in zip(*where(self.s<0)): # if stock becomes <=0, 0 out rest of path
            self.s[xx,yy:] = 0
        if self.call_put == 'call':
            y_raw = maximum(self.s[:,-1] - self.strike_price, 0)
        else: # put
            y_raw = maximum(self.strike_price - self.s[:,-1], 0)
        y_adj = y_raw * exp(-self.interest_rate * self.t_final)
        return y_adj
    
    def get_exact_value(self):
        """
        Get the fair price of a European call/put option.
        
        Return:
            float: fair price
        """
        denom = self.volatility * sqrt(self.t_final)
        decay = self.strike_price * exp(-self.interest_rate * self.t_final)
        if self.call_put == 'call':
            term1 = log(self.start_price / self.strike_price) + \
                    (self.interest_rate + self.volatility**2/2) * self.t_final
            term2 = log(self.start_price / self.strike_price) + \
                    (self.interest_rate - self.volatility**2/2) * self.t_final
            fp = self.start_price * norm.cdf(term1/denom) - decay * norm.cdf(term2/denom)
        elif self.call_put == 'put':
            term1 = log(self.strike_price / self.start_price) - \
                    (self.interest_rate - self.volatility**2/2) * self.t_final
            term2 = log(self.strike_price / self.start_price) - \
                    (self.interest_rate + self.volatility**2/2) * self.t_final
            fp = decay * norm.cdf(term1/denom) - self.start_price * norm.cdf(term2/denom)
        return fp
    
    def _spawn(self, level, sampler):
        return EuropeanOption(
            sampler = sampler,
            volatility = self.volatility,
            start_price = self.start_price,
            strike_price = self.strike_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put)
