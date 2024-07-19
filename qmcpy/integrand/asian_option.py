from ..discrete_distribution import DigitalNetB2
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError
from numpy import *
from ._option import Option


class AsianOption(Option):
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
    >>> y.mean()
    1.768...
    >>> level_dims = [2,4,8]
    >>> ac2_multilevel = AsianOption(DigitalNetB2(seed=7),multilevel_dims=level_dims)
    >>> levels_to_spawn = arange(ac2_multilevel.max_level+1)
    >>> ac2_single_levels = ac2_multilevel.spawn(levels_to_spawn)
    >>> yml = 0
    >>> for ac2_single_level in ac2_single_levels:
    ...     x = ac2_single_level.discrete_distrib.gen_samples(2**12)
    ...     level_est = ac2_single_level.f(x).mean()
    ...     yml += level_est
    >>> yml
    1.779...
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

        self.mean_type = mean_type.lower()
        if self.mean_type not in ['arithmetic','geometric']:
            raise ParameterError("mean_type must either 'arithmetic' or 'geometric'")
        
        super(AsianOption, self).__init__(sampler, volatility, start_price,
                                          strike_price, interest_rate, t_final,
                                          call_put, multilevel_dims, _dim_frac)
        self.true_measure = BrownianMotion(self.sampler, self.t_final, decomp_type=decomp_type)

    def _get_discounted_payoffs(self, stock_path, dimension):
        """
        Calculate the discounted payoff from the stock path. 
        
        Args:
            stock_path (ndarray): n samples by d dimension option prices at monitoring times
            dimension (int): number of dimensions
        
        Return:
            ndarray: n vector of discounted payoffs
        """
        if self.mean_type == 'arithmetic':
            avg = (self.start_price / 2. +
                   stock_path[:, :-1].sum(1) +
                   stock_path[:, -1] / 2.) / \
                float(dimension)
        elif self.mean_type == 'geometric':
            avg = exp((log(self.start_price) / 2. +
                       log(stock_path[:, :-1]).sum(1) +
                       log(stock_path[:, -1]) / 2.) /
                      float(dimension))
        if self.call_put == 'call':
            y_raw = maximum(avg - self.strike_price, 0)
        else: # put
            y_raw = maximum(self.strike_price - avg, 0)
        y_adj = y_raw * exp(-self.interest_rate * self.t_final)
        return y_adj

    def g(self, t):
        if self.parent:
            raise ParameterError('''
                Cannot evaluate an integrand with multilevel_dims directly,
                instead spawn some children and evaluate those.''')
        self.s_fine = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2.) *
            self.true_measure.time_vec + self.volatility * t)
        for xx,yy in zip(*where(self.s_fine<0)): # if stock becomes <=0, 0 out rest of path
            self.s_fine[xx,yy:] = 0
        y = self._get_discounted_payoffs(self.s_fine,self.d)
        if self.dim_fracs > 0:
            s_course = self.s_fine[:, int(self.dim_fracs - 1):: int(self.dim_fracs)]
            d_course = float(self.d) / self.dim_fracs
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
            dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
    
