import numpy as np
import scipy as sc
from ._integrand import Integrand
from ..true_measure import BrownianMotion

class AmericanOption(Integrand):
    def __init__(self, sampler=None, volatility=0.2, start_price=30.0, strike_price=32.0,
                 interest_rate=0.05, t_final=1, call_put='put',multilevel_dims=None,decomp_type='PCA', _dim_frac=0):
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        self.sampler = sampler
        self.true_measure = BrownianMotion(self.sampler,t_final,decomp_type=decomp_type)
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.t_final = t_final
        self.decomp_type = decomp_type
        self.call_put = call_put.lower()
        # handle single vs multilevel
        self.multilevel_dims = multilevel_dims
        if self.multilevel_dims is not None: # multi-level problem
            self.dim_fracs = array(
                [0]+ [float(self.multilevel_dims[i])/float(self.multilevel_dims[i-1]) 
                for i in range(1,len(self.multilevel_dims))],
                dtype=float)
            self.max_level = len(self.multilevel_dims)-1
            self.leveltype = 'fixed-multi'
            self.parent = True
            self.parameters += ['multilevel_dims']
        else: # single level problem
            self.dim_frac = _dim_frac
            self.leveltype = 'single'
            self.parent = False
            self.parameters += ['dim_frac']
        super(AmericanOption,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)   
        
    
    def _get_discounted_payoffs(self,stock_path,dimension):
        """
        Calculate the discounted payoff from the stock path. 
        
        Args:
            
        Return:
            ndarray: n vector of discounted payoffs
        """
        npaths = stock_path.shape[0]
        cashflow = np.zeros(npaths)
        basis=lambda x: np.reshape(np.tile(np.exp(-x/2),3),(x.size,3),'F')* np.vstack([np.ones(x.size),1-x,1-2*x+x*x/2]).T
        putpayoff = np.maximum(self.strike_price-stock_path,0)*np.reshape(np.tile(np.exp(-self.interest_rate*np.arange(self.t_final/dimension,self.t_final+self.t_final/dimension,self.t_final/dimension)),npaths),(npaths,dimension))
        cashflow = putpayoff[:,-1]
        exbound = np.zeros(dimension+1)
        exbound[-1] = self.strike_price
        for i in range(dimension-2, 0, -1):
            inmoney = np.where(putpayoff[:,i] >0)[0]
            if inmoney.size != 0:
                regmat = np.ones((inmoney.size,4))
                regmat[:,1:4] = basis(stock_path[inmoney,i]/self.start_price)
                sol = np.linalg.lstsq(regmat,cashflow[inmoney],rcond =None)
                hold = np.dot(regmat,sol[0])
                shouldex=inmoney[putpayoff[inmoney,i]>hold]
                if shouldex.size != 0:
                    cashflow[shouldex]=putpayoff[shouldex,i]      
                    exbound[i+1]=np.max(stock_path[shouldex,i])
        if self.start_price < self.strike_price:
            hold = np.mean(cashflow)
            putpayoff0 = self.strike_price-self.start_price
            if putpayoff0 > hold:
                 cashflow = putpayoff0
                 exbound[0] = self.strike_price-hold   
        return cashflow


    def g(self,t):
         if self.parent:
            raise ParameterError('''
                Cannot evaluate an integrand with multilevel_dims directly,
                instead spawn some children and evaluate those.''')
         self.s_fine = self.start_price * np.exp(
            (self.interest_rate - self.volatility ** 2 / 2.) *
            self.true_measure.time_vec + self.volatility * t)
         for xx,yy in zip(*np.where(self.s_fine<0)): # if stock becomes <=0, 0 out rest of path
            self.s_fine[xx,yy:] = 0
         y = self._get_discounted_payoffs(self.s_fine,self.d)
         if self.dim_frac > 0:
            s_course = self.s_fine[:, int(self.dim_frac - 1):: int(self.dim_frac)]
            d_course = int(float(self.d) / self.dim_frac)
            y_course = self._get_discounted_payoffs(s_course, d_course)
            y -= y_course
         return y
    
    def _spawn(self, level, sampler):
        breakpoint()
        return AmericanOption(
            sampler = sampler,
            volatility = self.volatility,
            start_price = self.start_price,
            strike_price = self.strike_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put,
            _dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
