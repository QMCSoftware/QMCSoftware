import numpy as np
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import *
from ._option import Option

class LookBackOption(Option):
                          
    def __init__(self, sampler, volatility=0.5, start_price=30.0, interest_rate=0.0,
                 t_final=1, call_put='put', multilevel_dims=None, 
                 _dim_frac=0,decomp_type='PCA'):      
        
        # Call super class's constructor to define this class's attributes
        super(LookBackOption, self).__init__(sampler, volatility, start_price,
                                             0, interest_rate, t_final,
                                             call_put, multilevel_dims, _dim_frac)
        
        # Finish defining the rest of the class's attributes
        self.true_measure = BrownianMotion(self.sampler, self.t_final, decomp_type=decomp_type)
    
    def _get_discounted_payoffs(self,stock_path,dimension):
        if self.call_put == 'call':
           y_raw = stock_path[:,-1] - np.minimum(np.min(stock_path, axis=1),self.start_price)
        else: # put
            y_raw = np.maximum(np.max(stock_path, axis=1),self.start_price) - stock_path[:,-1]
        y_adj = y_raw * np.exp(-self.interest_rate * self.t_final)
        return y_adj 
     
    def g(self, t):
        if self.parent:
            raise ParameterError('''
                Cannot evaluate an integrand with multilevel_dims directly,
                instead spawn some children and evaluate those.''')
        self.s_fine = self.start_price * np.exp((self.interest_rate
                                              - self.volatility**2 / 2.0)
                                              * self.true_measure.time_vec
                                              + self.volatility * t)
        for xx,yy in zip(*np.where(self.s_fine<0)): # if stock becomes <=0, 0 out rest of path
            self.s_fine[xx,yy:] = 0
        y = self._get_discounted_payoffs(self.s_fine, self.d)
        if self.dim_fracs > 0:
            s_coarse = self.s_fine[:, int(self.dim_fracs - 1):: int(self.dim_fracs)]
            d_coarse = float(self.d) / self.dim_fracs
            y_coarse = self._get_discounted_payoffs(s_coarse, d_coarse)
            y -= y_coarse
        return y
        
    def _spawn(self, level, sampler):            
        return LookBackOption(
            sampler = sampler,
            volatility = self.volatility,
            start_price = self.start_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put,
            _dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
    