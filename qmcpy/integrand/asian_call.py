from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError
from numpy import array, exp, log, maximum, repeat


class AsianCall(Integrand):

    parameters = ['volatility', 'start_price', 'strike_price',
                  'interest_rate','mean_type', 'dimensions', 'dim_fracs']
                          
    def __init__(self, measure, volatility=0.5, start_price=30, strike_price=35,\
                 interest_rate=0, mean_type='arithmetic', multi_level_dimensions=None):
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
        self.volatility = volatility
        self.start_price = start_price
        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.mean_type = mean_type.lower()
        if self.mean_type not in ['arithmetic', 'geometric']:
            raise ParameterError("mean_type must either 'arithmetic' or 'geometric'")
        if multi_level_dimensions:
            # multi-level problem
            self.dimensions = multi_level_dimensions
            self.dim_fracs = [0] + [float(self.dimensions[i])/self.dimensions[i-1] \
                for i in range(1,len(self.dimensions))]
            self.multilevel = True
        else:
            # single level problem
            self.dimensions = [self.measure.distribution.dimension]
            self.dim_fracs = [0]
        self.exercise_time = self.measure.time_vector[-1]
        super().__init__()        

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
            avg = (self.start_price / 2 +
                   stock_path[:, :-1].sum(1) +
                   stock_path[:, -1] / 2) / \
                dimension
        elif self.mean_type == 'geometric':
            avg = exp((log(self.start_price) / 2 +
                       log(stock_path[:, :-1]).sum(1) +
                       log(stock_path[:, -1]) / 2) /
                      dimension)
        y = maximum(avg - self.strike_price, 0) * \
            exp(-self.interest_rate * self.exercise_time)
        return y

    def g(self, x, l=0):
        """ See abstract method. """
        dim_frac = self.dim_fracs[l]
        dimension = self.dimensions[l]
        s_fine = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.measure.time_vector + self.volatility * x)
        y = self.get_discounted_payoffs(s_fine, dimension)
        if dim_frac > 0:
            s_course = s_fine[:, int(dim_frac - 1):: int(dim_frac)]
            d_course = dimension / dim_frac
            y_course = self.get_discounted_payoffs(s_course, d_course)
            y -= y_course
        return y
    
    def dim_at_level(self, l):
        """ See abstract method. """
        return self.dimensions[l]
