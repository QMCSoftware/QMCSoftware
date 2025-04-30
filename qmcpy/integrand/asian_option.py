from ..discrete_distribution import DigitalNetB2
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError
import numpy as np


class AsianOption(Integrand):
    """
    Asian financial option. 

    >>> ac = AsianOption(DigitalNetB2(4,seed=7))
    >>> ac
    AsianOption (Integrand Object)
        volatility      2^(-1)
        call_put        call
        start_price     30
        strike_price    35
        interest_rate   0
        mean_type       arithmetic
        dim_frac        0
    >>> x = ac.discrete_distrib.gen_samples(2**12)
    >>> y = ac.f(x)
    >>> y.mean().item()
    1.7707127229398623
    >>> level_dims = [2,4,8]
    >>> ac2_multilevel = AsianOption(DigitalNetB2(seed=7),multilevel_dims=level_dims)
    >>> levels_to_spawn = np.arange(ac2_multilevel.max_level+1)
    >>> ac2_single_levels = ac2_multilevel.spawn(levels_to_spawn)
    >>> yml = 0
    >>> for ac2_single_level in ac2_single_levels:
    ...     x = ac2_single_level.discrete_distrib.gen_samples(2**12)
    ...     level_est = ac2_single_level.f(x).mean()
    ...     yml += level_est
    >>> yml.item()
    1.7782019427424296
    """
                          
    def __init__(self, sampler, volatility=0.5, start_price=30., strike_price=35.,\
        interest_rate=0., t_final=1, call_put='call', mean_type='arithmetic', multilevel_dims=None, decomp_type='PCA', _dim_frac=0):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
            mean_type (string): 'arithmetic' or 'geometric' mean
            multilevel_dims (list of ints): list of dimensions at each level. 
                Leave as None for single-level problems
            _dim_frac (float): for internal use only, users should not set this parameter. 
        """
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate','mean_type']
        self.sampler = sampler
        self.true_measure = BrownianMotion(self.sampler,t_final,decomp_type=decomp_type)
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.t_final = t_final
        self.decomp_type = decomp_type
        self.call_put = call_put.lower()
        if self.call_put not in ['call','put']:
            raise ParameterError("call_put must be either 'call' or 'put'")
        self.mean_type = mean_type.lower()
        if self.mean_type not in ['arithmetic','geometric']:
            raise ParameterError("mean_type must either 'arithmetic' or 'geometric'")
        # handle single vs multilevel
        self.multilevel_dims = multilevel_dims
        if self.multilevel_dims is not None: # multi-level problem
            self.dim_fracs = np.array(
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
        super(AsianOption,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)    

    def _get_discounted_payoffs(self, stock_path, dimension):
        """
        Calculate the discounted payoff from the stock path. 
        
        Args:
            stock_path (np.ndarray): n samples by d dimension option prices at monitoring times
            dimension (int): number of dimensions
        
        Return:
            np.ndarray: n vector of discounted payoffs
        """
        if self.mean_type == 'arithmetic':
            avg = (self.start_price / 2. +
                   stock_path[:, :-1].sum(1) +
                   stock_path[:, -1] / 2.) / \
                float(dimension)
        elif self.mean_type == 'geometric':
            avg = np.exp((np.log(self.start_price) / 2. +
                       np.log(stock_path[:, :-1]).sum(1) +
                       np.log(stock_path[:, -1]) / 2.) /
                      float(dimension))
        if self.call_put == 'call':
            y_raw = np.maximum(avg - self.strike_price, 0)
        else: # put
            y_raw = np.maximum(self.strike_price - avg, 0)
        y_adj = y_raw * np.exp(-self.interest_rate * self.t_final)
        return y_adj

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
        y = self._get_discounted_payoffs(self.s_fine,self.d)
        if self.dim_frac > 0:
            s_course = self.s_fine[:, int(self.dim_frac - 1):: int(self.dim_frac)]
            d_course = float(self.d) / self.dim_frac
            y_course = self._get_discounted_payoffs(s_course, d_course)
            y -= y_course
        return y
    
    def _dimension_at_level(self, level):
        return self.d if self.multilevel_dims is None else self.multilevel_dims[level]
        
    def _spawn(self, level, sampler):            
        return AsianOption(
            sampler = sampler,
            volatility = self.volatility,
            start_price = self.start_price,
            strike_price = self.strike_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put,
            mean_type = self.mean_type,
            _dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
    
