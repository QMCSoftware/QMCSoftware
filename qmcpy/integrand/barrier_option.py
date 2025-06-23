from ..discrete_distribution import DigitalNetB2
from .abstract_integrand import AbstractIntegrand
from ..true_measure import BrownianMotion
from ..util import ParameterError
from numpy import *
from ._option import Option

class BarrierOption(Option):
    """
    Barrier Option. 

    >>> barrier_option = BarrierOption(DigitalNetB2(4,seed=7))
    >>> barrier_option
    BarrierOption (Integrand Object)
        volatility      0.200
        call_put        call
        start_price     30
        strike_price    35
        barrier_price   38
        interest_rate   0.050
        in_out          in
        dim_frac        0
    >>> x = barrier_option.discrete_distrib.gen_samples(2**12)
    >>> y = barrier_option.f(x)
    >>> y.mean()
    1.08...
    >>> level_dims = [2,4,8]
    >>> barrier_option_multilevel = BarrierOption(DigitalNetB2(seed=7),multilevel_dims=level_dims)
    >>> levels_to_spawn = arange(barrier_option_multilevel.max_level+1)
    >>> barrier_option_single_levels = barrier_option_multilevel.spawn(levels_to_spawn)
    >>> yml = 0
    >>> for barrier_option_single_level in barrier_option_single_levels:
    ...     x = barrier_option_single_level.discrete_distrib.gen_samples(2**12)
    ...     level_est = barrier_option_single_level.f(x).mean()
    ...     yml += level_est
    >>> yml
    1.10...
    """

    def __init__(self, sampler, volatility=0.2, start_price=30., strike_price=35., barrier_price = 38.,\
        interest_rate=0.05, t_final=1., call_put='call', in_out='in', multilevel_dims=None, decomp_type='PCA', _dim_frac=0):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            barrier_price (float): price at which the option activates/deactivates
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
            multilevel_dims (list of ints): list of dimensions at each level. 
                Leave as None for single-level problems
            _dim_frac (float): for internal use only, users should not set this parameter. 
        """
        super(BarrierOption, self).__init__(sampler, volatility, start_price,strike_price,
                                            interest_rate, t_final, call_put, multilevel_dims,
                                            _dim_frac)

        self.true_measure = BrownianMotion(self.sampler, t_final, decomp_type=decomp_type)

        if(barrier_price == start_price):
            raise ParameterError("Barrier Price must be greater or less than the start price. They can't be equal.")

        self.barrier_price = float(barrier_price)
        self.decomp_type = decomp_type
        self.in_out = in_out.lower()

        if self.in_out not in ['in', 'out']:
            raise ParameterError("in_out must be either 'in' or 'out'")
        
        if(self.start_price < self.barrier_price):
            self.up = True
        else:
            self.up = False
        
        if(self.in_out == 'in'):
            self.i = True
        else:
            self.i = False
        
        if(self.call_put == 'call'):
            self.call = True
        else:
            self.call = False

    def _get_discounted_payoffs(self,stock_path,dimension):
        """
        Calculate the discounted payoff from the stock path. 
        
        Args:
            stock_path (ndarray): n samples by d dimension option prices at monitoring times
            dimension (int): number of dimensions
        
        Return:
            ndarray: n vector of discounted payoffs
        """
        expected_stock = stock_path[:,dimension - 1]
        if(self.call):
            disc_payoff = expected_stock - self.strike_price
        else:
            disc_payoff = self.strike_price - expected_stock
        if(self.up and self.i):
            bar_flag = stock_path >= self.barrier_price
            bar_flag = bar_flag.sum(axis = 1) > 0
        elif(self.up and (self.i is False)):
            bar_flag = stock_path < self.barrier_price
            bar_flag = bar_flag.sum(axis = 1) == dimension
        elif((self.up is False) and self.i):
            bar_flag = stock_path <= self.barrier_price
            bar_flag = bar_flag.sum(axis = 1) > 0
        else:
            bar_flag = stock_path > self.barrier_price
            bar_flag = bar_flag.sum(axis = 1) == dimension
        disc_payoff = disc_payoff*bar_flag
        disc_payoff = maximum(zeros(disc_payoff.size), disc_payoff)
        return (disc_payoff * exp(-self.interest_rate * self.t_final))
    
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
            d_course = int(float(self.d) / self.dim_fracs)
            y_course = self._get_discounted_payoffs(s_course, d_course)
            y -= y_course
        return y
        
    def _spawn(self, level, sampler):            
        return BarrierOption(
            sampler = sampler,
            volatility = self.volatility,
            start_price = self.start_price,
            strike_price = self.strike_price,
            barrier_price = self.barrier_price,
            interest_rate = self.interest_rate,
            t_final = self.t_final,
            call_put = self.call_put,
            in_out = self.in_out,
            _dim_frac = self.dim_fracs[level] if hasattr(self,'dim_fracs') else 0)
            
