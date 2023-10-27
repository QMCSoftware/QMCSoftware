
import numpy as np
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..discrete_distribution import DigitalNetB2


class AmericanOption(Integrand):
                          
    def __init__(self, volatility=0.5, start_price=30., strike_price=35.,\
        interest_rate=0., drift=0, n=16, t_final=1, call_put='put', multilevel_dims=None, _dim_frac=0, observations=10):
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        # handle single vs multilevel
        self.multilevel_dims = multilevel_dims
        self.call_put=call_put
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
            self.sampler=sampler=Sobol(1)
        self.observations=observations
        self.t_final = t_final
        self.n=n
        self.start_price=start_price
        self.interest_rate = interest_rate
        self.t=np.linspace(0,self.t_final, self.observations)
        self.true_measure = BrownianMotion(self.sampler,t_final)
        self.volatility = volatility
        self.stock_values =self.start_price*np.exp((self.interest_rate-0.5*self.volatility**2)*self.t+self.volatility*self.true_measure.gen_samples(self.n))
        super(AmericanOption,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)    
        
    
    def get_discounted_payoffs(self):
        b=[]
        h = np.zeros((self.n, self.observations))
        y = np.zeros(self.n)
        exercise_time=self.t_final
        max_function = np.vectorize(lambda x: max(x, 0))
        payoff_tree = max_function(self.strike_price-self.stock_values)
        payoff_tree = payoff_tree.astype(np.float64)
        payoff_tree *=np.exp(-self.interest_rate*self.t)
        for j in range(self.observations):
            for i in range(self.n):
                in_money_values = payoff_tree[i, j:][payoff_tree[i, j:] != 0]
                if len(in_money_values) > 0:
                    mean = np.mean(in_money_values)
                    h[i, j] = mean
                else:
                    h[i, j] = 0
                    
            x=self.stock_values[:,j]
            y=h[:,j]
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            bound=(self.strike_price-c)/(m+1)
            b.append(bound)        
            
        for i in range (self.n):
            j=1
            while j<self.observations :
                if payoff_tree[i,j]<b[j]:
                    y[i]=payoff_tree[i,j]
                    break
                else:
                    y[i]=payoff_tree[i,self.observations-1]
                j=j+1
        return np.mean(y)
        
    
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
    






