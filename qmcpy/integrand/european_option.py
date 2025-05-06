from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError
import numpy as np
from scipy.stats import norm 


class EuropeanOption(Integrand):
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
    >>> print("%.4f"%y.mean())
    9.2084
    >>> eo = EuropeanOption(BrownianMotion(DigitalNetB2(4,seed=7),drift=1),call_put='put')
    >>> x = eo.discrete_distrib.gen_samples(2**12)
    >>> y = eo.f(x)
    >>> print("%.4f"%y.mean())
    9.1957
    >>> print("%.4f"%eo.get_exact_value())
    9.2115
    """
                          
    def __init__(self, sampler, volatility=0.5, start_price=30, strike_price=35,
        interest_rate=0, t_final=1, call_put='call'):
        """
        Args:
            sampler (AbstractDiscreteDistribution/AbstractTrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
            call_put (str): 'call' or 'put' option
        """
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        self.t_final = t_final
        self.sampler = sampler
        self.true_measure = BrownianMotion(self.sampler,t_final=self.t_final)
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.call_put = call_put.lower()
        if self.call_put not in ['call','put']:
            raise ParameterError("call_put must be either 'call' or 'put'")
        super(EuropeanOption,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)  

    def g(self, t):
        """ See abstract method. """
        self.s = self.start_price * np.exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.true_measure.time_vec + self.volatility * t)
        for xx,yy in zip(*np.where(self.s<0)): # if stock becomes <=0, 0 out rest of path
            self.s[xx,yy:] = 0
        if self.call_put == 'call':
            y_raw = np.maximum(self.s[:,-1] - self.strike_price, 0)
        else: # put
            y_raw = np.maximum(self.strike_price - self.s[:,-1], 0)
        y_adj = y_raw * np.exp(-self.interest_rate * self.t_final)
        return y_adj
    
    def get_exact_value(self):
        """
        Get the fair price of a European call/put option.
        
        Return:
            float: fair price
        """
        denom = self.volatility * np.sqrt(self.t_final)
        decay = self.strike_price * np.exp(-self.interest_rate * self.t_final)
        if self.call_put == 'call':
            term1 = np.log(self.start_price / self.strike_price) + \
                    (self.interest_rate + self.volatility**2/2) * self.t_final
            term2 = np.log(self.start_price / self.strike_price) + \
                    (self.interest_rate - self.volatility**2/2) * self.t_final
            fp = self.start_price * norm.cdf(term1/denom) - decay * norm.cdf(term2/denom)
        elif self.call_put == 'put':
            term1 = np.log(self.strike_price / self.start_price) - \
                    (self.interest_rate - self.volatility**2/2) * self.t_final
            term2 = np.log(self.strike_price / self.start_price) - \
                    (self.interest_rate + self.volatility**2/2) * self.t_final
            fp = decay * norm.cdf(term1/denom) - self.start_price * norm.cdf(term2/denom)
        return fp
    
    def _spawn(self, level, sampler):
        return EuropeanOption(
            sampler=sampler,
            volatility=self.volatility,
            start_price=self.start_price,
            strike_price=self.strike_price,
            interest_rate=self.interest_rate,
            t_final=self.t_final,
            call_put=self.call_put)
