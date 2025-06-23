from qmcpy import *
import numpy as np
import scipy as sc
from .abstract_integrand import AbstractIntegrand
from ..true_measure import BrownianMotion
from ..util import ParameterError, MethodImplementationError

class Option(AbstractIntegrand):
    """Option abstract class. DO NOT INSTANTIATE."""
    def __init__(self, sampler, volatility: float, start_price: float, strike_price: float,
                 interest_rate: float, t_final: float, call_put: str, multilevel_dims: list[int],
                 dim_frac: float):
        """
        Args:
            sampler: DiscreteDistribution or TrueMeasure
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t = 0
            strike_price (float): the call/put offer
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
            call_put (str): 'call' or 'put' option
            multilevel_dims (list of ints): list of dimensions at each level.
                Leave as None for single-level problems
            _dim_frac (float): for internal use only, users should not set this parameter.
        """
        self.sampler: list[int] = sampler
        self.parameters: list[str] = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.t_final: float = t_final
        self.true_measure = BrownianMotion(sampler, t_final)
        self.call_put: str = call_put
        if self.call_put not in ('call', 'put'):
            raise ParameterError("call_put must be either 'call' or 'put'")
        
        # Handle single vs multilevel
        self.multilevel_dims: list[int] = multilevel_dims
        if self.multilevel_dims is not None: # multi-level problem
            self.dim_fracs = np.array(
                [0] + [float(self.multilevel_dims[i])/float(self.multilevel_dims[i-1]) for i in range(1, len(self.multilevel_dims))],
                dtype=float
            )
            self.max_level: int = len(self.multilevel_dims) - 1
            self.leveltype: str = 'fixed-multi'
            self.parent: bool = True
            self.parameters.append('multilevel_dims') # self.parameters += ['multilevel_dims']
        else: # single level problem
            self.dim_fracs = dim_frac
            self.leveltype: str = 'single'
            self.parent: bool = False
            self.parameters.append('dim_fracs') # self.parameters += ['dim_frac']

        super(Option, self).__init__(dimension_indv=1, dimension_comb=1, parallel=False)

    def _get_discounted_payoffs(self):
        """ABSTRACT METHOD"""
        raise MethodImplementationError(self, '_get_discounted_payoffs')

    def _dimension_at_level(self, level):
        return self.d if self.multilevel_dims is None else self.multilevel_dims[level]