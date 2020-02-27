""" Definition for class AsianCall, a concrete implementation of Integrand """

from numpy import array, exp, log, maximum

from ._integrand import Integrand
from ..util import ParameterError


class AsianCall(Integrand):
    """ Specify and generate payoff values of an Asian Call option """

    def __init__(self, bm_measure, volatility=0.5, start_price=30,
                 strike_price=25, interest_rate=0, mean_type='arithmetic'):
        """
        Initialize AsianCall Integrand(s)

        Args:
            bm_measure (TrueMeasure): A BrownianMotion Measure object
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            mean_type (string): 'arithmetic' or 'geometric' mean
        """
        mean_type = mean_type.lower()
        if mean_type not in ['arithmetic', 'geometric']:
            raise ParameterError("mean_type must either 'arithmetic' or 'geometric'")
        dimension = bm_measure.dimension
        super().__init__(dimension,
                         bm_measure=bm_measure,
                         dim_frac=array([0] + [dimension[i] / dimension[i - 1]
                                               for i in range(1, len(dimension))]),
                         volatility=volatility,
                         start_price=start_price,
                         strike_price=strike_price,
                         interest_rate=interest_rate,
                         mean_type=[mean_type],
                         exercise_time=[bm_measure[i].time_vector[-1] for i in range(len(dimension))])

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
            self.bm_measure.time_vector + self.volatility * x)
        y = self.get_discounted_payoffs(s_fine, self.dimension)
        if self.dim_frac > 0:
            s_course = s_fine[:, int(self.dim_frac - 1):: int(self.dim_frac)]
            d_course = self.dimension / self.dim_frac
            y_course = self.get_discounted_payoffs(s_course, d_course)
            y -= y_course
        return y

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

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        attributes = ['volatility', 'start_price', 'strike_price',
                      'interest_rate', 'mean_type', 'exercise_time']
        return super().__repr__(attributes)
