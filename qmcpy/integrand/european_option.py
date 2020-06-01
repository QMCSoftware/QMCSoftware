from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError
from numpy import exp, maximum, log, sqrt
from scipy.stats import norm 


class EuropeanOption(Integrand):

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
        self.volatility = volatility
        self.start_price = start_price
        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.call_put = call_put.lower()
        if self.call_put not in ['call','put']:
            raise ParameterError("call_put must be either 'call' or 'put'")
        self.exercise_time = self.measure.time_vector[-1]
        super().__init__()        

    def g(self, x):
        """ See abstract method. """
        s = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.measure.time_vector + self.volatility * x)
        if self.call_put == 'call':
            y = maximum(s[:,-1] - self.strike_price, 0) * \
                exp(-self.interest_rate * self.exercise_time)
        else: # put
            s = maximum(s,0) # avoid negative stock values
            y = maximum(self.strike_price - s[:,-1], 0) * \
                exp(-self.interest_rate * self.exercise_time)
        return y
    
    def get_fair_price(self):
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
