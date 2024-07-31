import numpy as np
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import *
from ._option import Option

class LookBackOption(Option):
                          
    def __init__(self, sampler, volatility=0.5, start_price=30.0, interest_rate=0.0,
                 drift=0, n=16, t_final=1, call_put='put', multilevel_dims=None,
                 _dim_frac=0, observations=10):      
        self.observations = observations
        self.n = n # Number of samples
        
        # Call super class's constructor to define this class's attributes
        super(LookBackOption, self).__init__(sampler, volatility, start_price,
                                             0, interest_rate, t_final,
                                             call_put, multilevel_dims, _dim_frac)
        
        # Finish defining the rest of the class's attributes
        self.t = np.linspace(0, self.t_final, self.observations)
        self.true_measure = BrownianMotion(self.sampler, t_final)
        
    def g(self, t):
        """See abstract method."""
        s = self.start_price * np.exp((self.interest_rate
                                       - self.volatility**2 / 2.0)
                                       * self.true_measure.time_vec
                                       + self.volatility * t)
        for xx, yy in zip(*np.where(s < 0)):
            s[xx, yy:] = 0
        if self.call_put == 'call':
            y_raw = s[:,-1] - np.min(s, axis=1)
        else: # put
            y_raw = np.max(s, axis=1) - s[:,-1]
        y_adj = y_raw * np.exp(-self.interest_rate * self.t_final)
        return y_adj

    # def _get_discounted_payoffs(self):
    #     payoffs = []
    #     if self.call_put == 'call':
    #         for i in range(self.n):
    #             strike_price = min(self.stock_values[i,:])
    #             payoff = self.stock_values[i,self.observations-1] - strike_price
    #             payoffs.append(payoff)
    #     else: # put
    #         for i in range(self.n):
    #             strike_price = max(self.stock_values[i,:])
    #             payoff = strike_price - self.stock_values[i,self.observations-1]
    #             payoffs.append(payoff)
    #     return np.mean(payoffs)
    
    # def get_discounted_payoffs(self):
    #     """
    #     Wrapper method for the private _get_discounted_payoffs method.
    #     Although the user could just call the original method directly,
    #     it's better to stick with Python convention and not call the
    #     private method directly, but instead through a wrapper. The
    #     reason for this is we still want to user to be able use this,
    #     while also overriding the abstract method as well.
    #     """
    #     return self._get_discounted_payoffs()
        
    def _spawn(self, level, sampler):            
        return LookBackOption(
            sampler = sampler,
            volatility = self.volatility,
            start_price = self.start_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put,
            _dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
    