from ..discrete_distribution import Sobol
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError
from numpy import *


class AsianOption(Integrand):
    """
    Asian financial option. 

    >>> ac = AsianOption(Sobol(4,seed=7))
    >>> ac
    AsianOption (Integrand Object)
        volatility      2^(-1)
        call_put        call
        start_price     30
        strike_price    35
        interest_rate   0
        mean_type       arithmetic
        dimensions      2^(2)
        dim_fracs       0
    >>> x = ac.discrete_distrib.gen_samples(2**10)
    >>> y = ac.f(x)
    >>> y.mean()
    1.774...
    >>> level_dims = [2,4,8]
    >>> ac2 = AsianOption(Sobol(seed=7),multi_level_dimensions=level_dims)
    >>> ac2
    AsianOption (Integrand Object)
        volatility      2^(-1)
        call_put        call
        start_price     30
        strike_price    35
        interest_rate   0
        mean_type       arithmetic
        dimensions      [2 4 8]
        dim_fracs       [0. 2. 2.]
    >>> y2 = 0
    >>> for l in range(len(level_dims)):
    ...     new_dim = ac2._dim_at_level(l)
    ...     ac2.true_measure._set_dimension_r(new_dim)
    ...     x2 = ac2.discrete_distrib.gen_samples(2**10)
    ...     level_est = ac2.f(x2,l=l).mean()
    ...     y2 += level_est
    >>> y2
    1.772...
    """
                          
    def __init__(self, sampler, volatility=0.5, start_price=30., strike_price=35.,\
                 interest_rate=0., t_final=1, call_put='call', mean_type='arithmetic', multi_level_dimensions=None):
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
            multi_level_dimensions (list of ints): list of dimensions at each level. 
                Leave as None for single-level problems
        """
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', \
                  'interest_rate','mean_type', 'dimensions', 'dim_fracs']
        self.t_final = t_final
        self.true_measure = BrownianMotion(sampler,self.t_final)
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.call_put = call_put.lower()
        if self.call_put not in ['call','put']:
            raise ParameterError("call_put must be either 'call' or 'put'")
        self.mean_type = mean_type.lower()
        if self.mean_type not in ['arithmetic', 'geometric']:
            raise ParameterError("mean_type must either 'arithmetic' or 'geometric'")
        if multi_level_dimensions:
            # multi-level problem
            self.dimensions = multi_level_dimensions
            self.dim_fracs = [0.] + [float(self.dimensions[i])/float(self.dimensions[i-1]) \
                for i in range(1,len(self.dimensions))]
            self.leveltype = 'fixed-multi'
        else:
            # single level problem
            self.dimensions = [self.true_measure.d]
            self.dim_fracs = [0.]
            self.leveltype = 'single'
        self.dprime = 1
        super(AsianOption,self).__init__()    

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

    def g(self, t, l=0):
        """ See abstract method. """
        dim_frac = self.dim_fracs[l]
        dimension = float(self.dimensions[l])
        self.s_fine = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2.) *
            self.true_measure.time_vec + self.volatility * t)
        for xx,yy in zip(*where(self.s_fine<0)): # if stock becomes <=0, 0 out rest of path
            self.s_fine[xx,yy:] = 0
        y = self._get_discounted_payoffs(self.s_fine, dimension)
        if dim_frac > 0:
            s_course = self.s_fine[:, int(dim_frac - 1):: int(dim_frac)]
            d_course = float(dimension) / dim_frac
            y_course = self._get_discounted_payoffs(s_course, d_course)
            y -= y_course
        return y
    
    def _dim_at_level(self, l):
        """ See abstract method. """
        return self.dimensions[l]
