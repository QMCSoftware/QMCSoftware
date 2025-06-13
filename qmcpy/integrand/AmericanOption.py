import numpy as np
from qmcpy import *
import matplotlib.pyplot as plt
import scipy as sc
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ._option import Option

class AmericanOption(Option):
    def __init__(self, sampler=None, volatility=0.2, start_price=30.0, strike_price=32.0,
                 interest_rate=0.05, n=4096, t_final=1, call_put='put',
                 multilevel_dims=None, _dim_frac=0, observations=52):
        self.observations = observations # number of time steps KEEP
        self.n = n # Number of samples, Note, this will need to be removed.
        self.t = np.linspace(1/self.observations, t_final, self.observations)
        super(AmericanOption, self).__init__(sampler, volatility, start_price,
                                             strike_price, interest_rate, t_final,
                                             call_put, multilevel_dims, _dim_frac)
        self.stock_values = None
        
    
    def _get_discounted_payoffs(self):
        self.stock_values = self.start_price * np.exp((self.interest_rate - 0.5
                                                       * self.volatility**2) * self.t + self.volatility
                                                       * self.true_measure.gen_samples(self.n))
        in_money_values = np.zeros(self.n, dtype=bool)
        y = np.zeros(self.n) # Output for linear regression
        exercise_time = np.ones(self.n)
        exercise_time = self.t_final * exercise_time
        max_function = np.vectorize(lambda x: max(x, 0))
        payoff = max_function(self.strike_price - self.stock_values) # Payoffs at each time (or values assuming we do not exercise at any time)
        payoff = payoff.astype(np.float64)
        payoff *= np.exp(-self.interest_rate * self.t)
        values = payoff[:,self.observations-1] # Option values after comparing payoffs with linear regression
        #print("values",values)
        for j in range(self.observations-1, 0, -1):
            in_money_values = payoff[:,j]>0
            x = self.stock_values[in_money_values,j-1]
            y = values[in_money_values]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0] # Use multilinear regression
            hold_lr = m * self.stock_values[in_money_values,j-1] + c # Lr of holding value one more time step
            exercise = np.copy(in_money_values)
            exercise[in_money_values] = payoff[in_money_values,j-1] > hold_lr
            values[exercise] = payoff[exercise,j-1]
            exercise_time[exercise] = self.t[j-1]
        return values

    def g(self, t):
        self.n = len(t)
        if self.parent:
            raise ParameterError('''
                Cannot evaluate an integrand with multilevel_dims directly,
                instead spawn some children and evaluate those.''')
        self.s_fine = self.start_price * np.exp(
            (self.interest_rate - self.volatility ** 2 / 2.0)
            * self.true_measure.time_vec + self.volatility * t
        )
        for xx,yy in zip(*np.where(self.s_fine<0)): # if stock becomes <=0, 0 out rest of path
            self.s_fine[xx,yy:] = 0
        y = self._get_discounted_payoffs()
        if self.dim_fracs > 0:
            s_coarse = self.s_fine[:, int(self.dim_fracs - 1):: int(self.dim_fracs)]
            d_coarse = float(self.d) / self.dim_fracs
            y_coarse = self._get_discounted_payoffs()
            y -= y_coarse
        return y
    
    def _spawn(self, level, sampler):
        return AmericanOption(
            sampler = sampler,
            volatility = self.volatility,
            start_price = self.start_price,
            strike_price = self.strike_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put,
            _dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
