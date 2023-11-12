
import numpy as np
from qmcpy import *
import matplotlib.pyplot as plt
import scipy as sc

class AmericanOption(Integrand):
                          
    def __init__(self, volatility=0.2, start_price=30., strike_price=32.,\
        interest_rate=0.05, n=4096, t_final=1, call_put='put', multilevel_dims=None, _dim_frac=0, observations=52):
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        # handle single vs multilevel
        self.multilevel_dims = multilevel_dims
        self.call_put=call_put
        self.observations=observations #number of time steps
        self.strike_price=strike_price
        if self.multilevel_dims is not None: # multi-level problem
            self.dim_fracs = array(
                [0]+ [float(self.multilevel_dims[i])/float(self.multilevel_dims[i-1]) 
                for i in range(1,len(self.multilevel_dims))],
                dtype=float)
            self.max_level = len(self.multilevel_dims)-1
            self.leveltype = 'fixed-multi'
            self.parent = True
            self.parameters += ['multilevel_dims']
            self.sampler=Sobol(self.multilevel_dims),
        else: # single level problem
            self.dim_fracs = _dim_frac
            self.leveltype = 'single'
            self.parent = False
            self.parameters += ['dim_frac']
            self.sampler=sampler=Sobol(self.observations)
        self.t_final = t_final
        self.n=n
        self.start_price=start_price
        self.interest_rate = interest_rate
        self.t=np.linspace(1/self.observations,self.t_final, self.observations)
        self.true_measure = BrownianMotion(self.sampler,self.t_final) #Browninan motion
        self.volatility = volatility
        super(AmericanOption,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)    
        
    
    def _get_discounted_payoffs(self):
        self.stock_values =self.start_price*np.exp((self.interest_rate-0.5*self.volatility**2)*self.t+self.volatility*self.true_measure.gen_samples(self.n))
        in_money_values=np.zeros(self.n,dtype=bool)
        y = np.zeros(self.n) #output for linear regression
        exercise_time=np.ones(self.n)
        exercise_time=self.t_final*exercise_time
        max_function = np.vectorize(lambda x: max(x, 0))
        payoff = max_function(self.strike_price-self.stock_values) #Payoffs at each time (or values assuming we do not exercise at any time)
        payoff = payoff.astype(np.float64)
        payoff *=np.exp(-self.interest_rate*self.t)
        values=payoff[:,self.observations-1] #Option values after comparing payoffs with linear regression
        #print("values",values)
        for j in range(self.observations-1,0,-1):
            in_money_values=payoff[:,j]>0
            x=self.stock_values[in_money_values,j-1]
            y=values[in_money_values]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0] #use multilinear regression
            hold_lr=m*self.stock_values[in_money_values,j-1]+c #lr of holding value one more time step
            exercise=in_money_values
            exercise[in_money_values]=payoff[in_money_values,j-1]>hold_lr
            values[exercise]=payoff[exercise,j-1]
            exercise_time[exercise]=self.t[j-1]
        return values

    def g(self, t):
        if self.parent:
            raise ParameterError('''
                Cannot evaluate an integrand with multilevel_dims directly,
                instead spawn some children and evaluate those.''')
        self.s_fine = self.start_price * np.exp(
            (self.interest_rate - self.volatility ** 2 / 2.) *
            self.true_measure.time_vec + self.volatility * t)
        for xx,yy in zip(*np.where(self.s_fine<0)): # if stock becomes <=0, 0 out rest of path
            self.s_fine[xx,yy:] = 0
        y = self._get_discounted_payoffs()
        if self.dim_fracs > 0:
            s_course = self.s_fine[:, int(self.dim_fracs - 1):: int(self.dim_fracs)]
            d_course = float(self.d) / self.dim_fracs
            y_course = self._get_discounted_payoffs()
            y -= y_course
        return y
    
    
    def _dimension_at_level(self, level):
        return self.d if self.multilevel_dims is None else self.multilevel_dims[level]
        
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
    






