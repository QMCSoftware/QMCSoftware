from ..discrete_distribution import Sobol
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError
from numpy import array, exp, log, maximum, repeat


class AsianCall(Integrand):
    """
    >>> dd = Sobol(4,seed=7)
    >>> m = BrownianMotion(dd)
    >>> ac = AsianCall(m)
    >>> ac
    AsianCall (Integrand Object)
        volatility      0.5000
        start_price     30
        strike_price    35
        interest_rate   0
        mean_type       arithmetic
        dimensions      4
        dim_fracs       0
    >>> x = dd.gen_samples(2**10)
    >>> y = ac.f(x)
    >>> y.mean()
    1.7638343801580052

    >>> dd2 = Sobol(seed=7)
    >>> m2 = BrownianMotion(dd2,drift=1)
    >>> level_dims = [2,4,8]
    >>> ac2 = AsianCall(m2,multi_level_dimensions=level_dims)
    >>> ac2
    AsianCall (Integrand Object)
        volatility      0.5000
        start_price     30
        strike_price    35
        interest_rate   0
        mean_type       arithmetic
        dimensions      [2 4 8]
        dim_fracs       [ 0.000  2.000  2.000]
    >>> y2 = 0
    >>> for l in range(len(level_dims)):
    ...     new_dim = ac2.dim_at_level(l)
    ...     m2.set_dimension(new_dim)
    ...     x2 = dd2.gen_samples(2**10)
    ...     y2 += ac2.f(x2,l=l).mean()
    >>> y2
    1.787834256519869
    """

    parameters = ['volatility', 'start_price', 'strike_price',
                  'interest_rate','mean_type', 'dimensions', 'dim_fracs']
                          
    def __init__(self, measure, volatility=0.5, start_price=30., strike_price=35.,\
                 interest_rate=0., mean_type='arithmetic', multi_level_dimensions=None):
        """
        Args:
            measure (TrueMeasure): A BrownianMotion TrueMeasure object
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            mean_type (string): 'arithmetic' or 'geometric' mean
            multi_level_dimensions (list of ints): list of dimensions at each level. 
                Leave as None for single-level problems
        """
        if not isinstance(measure,BrownianMotion):
            raise ParameterError('AsianCall measure must be a BrownianMotion instance')
        self.measure = measure
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.mean_type = mean_type.lower()
        if self.mean_type not in ['arithmetic', 'geometric']:
            raise ParameterError("mean_type must either 'arithmetic' or 'geometric'")
        if multi_level_dimensions:
            # multi-level problem
            self.dimensions = multi_level_dimensions
            self.dim_fracs = [0.] + [float(self.dimensions[i])/float(self.dimensions[i-1]) \
                for i in range(1,len(self.dimensions))]
            self.multilevel = True
        else:
            # single level problem
            self.dimensions = [self.measure.distribution.dimension]
            self.dim_fracs = [0.]
        self.exercise_time = self.measure.time_vector[-1]
        super(AsianCall,self).__init__()        

    def get_discounted_payoffs(self, stock_path, dimension):
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
        y = maximum(avg - self.strike_price, 0.) * \
            exp(-self.interest_rate * self.exercise_time)
        return y

    def g(self, x, l=0):
        """ See abstract method. """
        dim_frac = self.dim_fracs[l]
        dimension = float(self.dimensions[l])
        s_fine = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2.) *
            self.measure.time_vector + self.volatility * x)
        y = self.get_discounted_payoffs(s_fine, dimension)
        if dim_frac > 0:
            s_course = s_fine[:, int(dim_frac - 1):: int(dim_frac)]
            d_course = float(dimension) / dim_frac
            y_course = self.get_discounted_payoffs(s_course, d_course)
            y -= y_course
        return y
    
    def dim_at_level(self, l):
        """ See abstract method. """
        return self.dimensions[l]
