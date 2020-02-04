""" Definition for class AsianCall, a concrete implementation of Integrand """

from ._integrand import Integrand
from ..measure._measure import Measure
from ..measure import BrownianMotion
from ..util import ParameterError
from numpy import array, exp, log, maximum, repeat


class AsianCall(Integrand):
    """ Specify and generate payoff values of an Asian Call option """

    parameters = ['volatility', 'start_price', 'strike_price',
                  'interest_rate','mean_type', 'dim_frac']
                          
    def __init__(self, measure, volatility=0.5, start_price=30, strike_price=25,\
                 interest_rate=0, mean_type='arithmetic', dim_frac=0):
        """
        Initialize AsianCall Integrand's'

        Args:
            measure (Measure): A BrownianMotion Measure object
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            mean_type (string): 'arithmetic' or 'geometric' mean
            dim_frac (float): fraciton of dimension compared to next level.
                              0 for single-level problems.
                              See add_multilevel_kwargs
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
        self.dim_frac = dim_frac
        self.dimension = self.measure.dimension
        self.exercise_time = self.measure.time_vector[-1]
        super().__init__()

    @classmethod
    def add_multilevel_kwargs(cls, levels, **kwargs):
        """ CLASS METHOD
        Add keyword argumnets to be distributed across levels
        
        Args:
            levels (int): number of levels 
            kwargs (dict): dictionary of keyword and multi-level arguments
                key (string): keyword
                val (list/ndarray): list of length levels whose elements will be 
                                    distributed amongst levels
        
        Returns: 
            kwargs (dict): input kwargs updated with more arguments
        """
        dims = [measure.dimension for measure in kwargs['measure']]
        kwargs['dim_frac'] = [0] + [dims[i]/dims[i-1] for i in range(1,levels)]
        return kwargs

    def get_discounted_payoffs(self, stock_path, dimension):
        """
        Calculate the discounted payoff from the stock path

        stock_path (ndarray): option prices at monitoring times
        dimension (int): number of dimensions
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

    def g(self, x):
        """
        Original integrand to be integrated

        Args:
            x: nodes, :math:`\\boldsymbol{x}_{\\mathfrak{u},i} = i^{\\mathtt{th}}` \
                row of an :math:`n \\cdot |\\mathfrak{u}|` matrix

        Returns:
            :math:`n \\cdot p` matrix with values \
            :math:`f(\\boldsymbol{x}_{\\mathfrak{u},i},\\mathbf{c})` where if \
            :math:`\\boldsymbol{x}_i' = (x_{i, \\mathfrak{u}},\\mathbf{c})_j`, \
            then :math:`x'_{ij} = x_{ij}` for :math:`j \\in \\mathfrak{u}`, \
            and :math:`x'_{ij} = c` otherwise
        """
        s_fine = self.start_price * exp(
            (self.interest_rate - self.volatility ** 2 / 2) *
            self.measure.time_vector + self.volatility * x)
        y = self.get_discounted_payoffs(s_fine, self.dimension)
        if self.dim_frac > 0:
            s_course = s_fine[:, int(self.dim_frac - 1):: int(self.dim_frac)]
            d_course = self.dimension / self.dim_frac
            y_course = self.get_discounted_payoffs(s_course, d_course)
            y -= y_course
        return y
