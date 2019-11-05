""" Definition for class AsianCall, a concrete implementation of Integrand """

from numpy import exp, maximum

from ._integrand import Integrand
from .._util import ParameterError


class AsianCall(Integrand):
    """ Specify and generate payoff values of an Asian Call option """

    def __init__(self, bm_measure=None, volatility=0.5, start_price=30, \
                 strike_price=25, interest_rate=0, mean_type='Arithmetic'):
        """
        Initialize AsianCall Integrand's'

        Args:
            bm_measure (TrueMeasure): A BrownianMotion Measure object
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t=0
            strike_price (float): strike_price, the call/put offer
            interest_rate (float): r, the annual interest rate
            mean_type (string): 'arithmetic' or 'geometric' mean
        """
        super().__init__()
        self.bm_measure = bm_measure
        self.volatility = volatility
        self.start_price = start_price
        self.strike_price = strike_price
        self.interest_rate = interest_rate
        self.mean_type = mean_type.lower()
        if self.mean_type not in ['arithmetic','geometric']:
            raise ParameterError("mean_type must either 'arithmetic' or 'geometric'")
        self.dim_fac = 0
        if not self.bm_measure: return
        # Create a list of Asian Call Options and distribArithmeticute attributes
        n_bm = len(bm_measure)
        self.integrand_list = [AsianCall() for i in range(n_bm)]
        for i in range(n_bm):  # distribute attr
            self[i].bm_measure = self.bm_measure[i]
            self[i].dim_fac = 0 if i == 0 else self.bm_measure[i].dimension / \
                self.bm_measure[i - 1].dimension
            self[i].dimension = self.bm_measure[i].dimension
            self[i].T = self.bm_measure[i].time_vector[-1]

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
            (self.interest_rate - self.volatility ** 2 / 2) * \
            self.bm_measure.time_vector + self.volatility * x)
        if self.mean_type == 'arithmetic':
            avg_fine = (self.start_price/2 + s_fine[:,:-1].sum(1) + s_fine[:,-1]/2) / \
                        self.dimension
            # Shouldn't this be: avg_fine = s_fine.mean(1) ???
        elif self.mean_type == 'geometric':
            # Should be: avg_fine = s_fine.prod(1)**(1./s_fine.shape[1])
            raise Exception('Geometric mean not yet implemented')
        y = maximum(avg_fine - self.strike_price, 0) * exp(-self.interest_rate*self.T)
        if self.dim_fac > 0:
            scourse = s_fine[:, int(self.dim_fac - 1):: int(self.dim_fac)]
            d_course = self.dimension / self.dim_fac
            avg_course = ((self.start_price / 2)
                          + scourse[:, : int(d_course) - 1].sum(1)
                          + scourse[:, int(d_course) - 1] / 2) / d_course
            y -= maximum(avg_course - self.strike_price, 0)* exp(-self.interest_rate*self.T)
        return y
