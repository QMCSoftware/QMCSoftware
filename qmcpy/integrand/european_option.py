from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import Sobol
from ..util import ParameterError
from numpy import exp, maximum, log, sqrt
from scipy.stats import norm 


class EuropeanOption(Integrand):
    """
    >>> dd = Sobol(4,seed=7)
    >>> m = BrownianMotion(dd,drift=-1)
    >>> eo = EuropeanOption(m,call_put='put')
    >>> eo
    EuropeanOption (Integrand Object)
        volatility      2^(-1)
        start_price     30
        strike_price    35
        interest_rate   0
    >>> x = dd.gen_samples(2**10)
    >>> y = eo.f(x)
    >>> y.mean()
    9.211371880941195
    """

    parameters = ['volatility', 'start_price', 'strike_price', 'interest_rate']
                          
    def __init__(self, measure, volatility=0.5, start_price=30, strike_price=35,
        interest_rate=0, call_put='call'):
        """
        Args:
            measure (TrueMeasure): A BrownianMotion TrueMeasure object
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            call_put (str): 'call' or 'put' option
        """
        if not isinstance(measure,BrownianMotion):
            raise ParameterError('EuropeanCall measure must be a BrownianMotion instance')
        self.measure = measure
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.call_put = call_put.lower()
        if self.call_put not in ['call','put']:
            raise ParameterError("call_put must be either 'call' or 'put'")
        self.exercise_time = self.measure.time_vector[-1]
        super(EuropeanOption,self).__init__()        

    def g(self, x):
        """ See abstract method. """
        s_last = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.measure.time_vector[-1] + self.volatility * x[:,-1])
        if self.call_put == 'call':
            y_raw = maximum(s_last - self.strike_price, 0)
        else: # put
            s_last = maximum(s_last,0) # avoid negative stock values
            y_raw = maximum(self.strike_price - s_last, 0)
        y_adj = y_raw * exp(-self.interest_rate * self.exercise_time)
        return y_adj
    
    def get_exact_value(self):
        """
        Get the fair price of a European call/put option 
        under geometric Brownain Motion.
        
        Return:
            float: fair price
        """
        denom = self.volatility * sqrt(self.exercise_time)
        decay = self.strike_price * exp(-self.interest_rate * self.exercise_time)
        if self.call_put == 'call':
            term1 = log(self.start_price / self.strike_price) + \
                    (self.interest_rate + self.volatility**2/2) * self.exercise_time
            term2 = log(self.start_price / self.strike_price) + \
                    (self.interest_rate - self.volatility**2/2) * self.exercise_time
            fp = self.start_price * norm.cdf(term1/denom) - decay * norm.cdf(term2/denom)
        elif self.call_put == 'put':
            term1 = log(self.strike_price / self.start_price) - \
                    (self.interest_rate - self.volatility**2/2) * self.exercise_time
            term2 = log(self.strike_price / self.start_price) - \
                    (self.interest_rate + self.volatility**2/2) * self.exercise_time
            fp = decay * norm.cdf(term1/denom) - self.start_price * norm.cdf(term2/denom)
        return fp 
