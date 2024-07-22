
import numpy as np
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import *

class LookBackOption(Integrand):
                          
    def __init__(self, volatility=0.5, start_price=30.,\
        interest_rate=0., drift=0, n=16, t_final=1, call_put='put', multilevel_dims=None, _dim_frac=0, observations=10):
        self.parameters = ['volatility', 'call_put', 'start_price', 'interest_rate']
        # handle single vs multilevel
        self.multilevel_dims = multilevel_dims
        self.call_put=call_put
        if self.multilevel_dims is not None: # multi-level problem
            self.dim_fracs = np.array(
                [0]+ [float(self.multilevel_dims[i])/float(self.multilevel_dims[i-1]) 
                for i in range(1,len(self.multilevel_dims))],
                dtype=float)
            self.max_level = len(self.multilevel_dims)-1
            self.leveltype = 'fixed-multi'
            self.parent = True
            self.parameters += ['multilevel_dims']
            self.sampler=Sobol(self.multilevel_dims),
        else: # single level problem
            self.dim_frac = _dim_frac
            self.leveltype = 'single'
            self.parent = False
            self.parameters += ['dim_frac']
            self.sampler=sampler=Sobol(1)
        self.observations=observations
        self.t_final = t_final # Exercise time
        self.n=n # Number of samples
        self.start_price=start_price
        self.interest_rate = interest_rate
        self.t=np.linspace(0,self.t_final, self.observations)
        self.true_measure = BrownianMotion(self.sampler,t_final)
        self.volatility = volatility
        self.stock_values =self.start_price*np.exp((self.interest_rate-0.5*self.volatility**2)*self.t+self.volatility*self.true_measure.gen_samples(self.n))
        super(LookBackOption,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)    
        
    def g(self, t):
        """See abstract method."""
        self.n = len(t)
        self.stock_values = self.start_price * np.exp((self.interest_rate - 0.5 * self.volatility**2)
                                                      * self.t + self.volatility * self.true_measure.gen_samples(self.n))
        self.s = self.start_price * np.exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.true_measure.time_vec + self.volatility * t)
        for xx, yy in zip(*np.where(self.s < 0)):
            self.s[xx, yy:] = 0
        if self.call_put == 'call':
            y_raw = self.s[:,-1] - np.min(self.s)
        else: # put
            y_raw = np.max(self.s) - self.s[:,-1]
        y_adj = y_raw * np.exp(-self.interest_rate * self.t_final)
        return y_adj

    def get_discounted_payoffs(self):
        payoffs=[]
        if self.call_put=='call':
            for i in range(self.n):
                strike_price=min(self.stock_values[i,:])
                payoff=self.stock_values[i,self.observations-1]-strike_price
                payoffs.append(payoff)
        else:
            for i in range(self.n):
                strike_price=max(self.stock_values[i,:])
                payoff=strike_price-self.stock_values[i,self.observations-1]
                payoffs.append(payoff)
        return np.mean(payoffs)
            
    
    def _dimension_at_level(self, level):
        return self.d if self.multilevel_dims is None else self.multilevel_dims[level]
        
    def _spawn(self, level, sampler):            
        return LookBackOption(
            volatility = self.volatility,
            start_price = self.start_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put,
            _dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
    