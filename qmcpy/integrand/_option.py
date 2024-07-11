from qmcpy import *
import numpy as np
import scipy as sc
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError, MethodImplementationError

class Option(Integrand):
    """
    Option abstract class. DO NOT INSTANTIATE.
    Date Created: 7/10/2024
    Author: Richard E. Varela
    """

    def __init__(self, sampler:BrownianMotion, volatility:float, start_price:float,\
                 strike_price:float, interest_rate:float, t_final, call_put:str,\
                 multilevel_dims, dim_frac:float):
        """
        Args:
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
        self.sampler = sampler
        self.parameters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        self.volatility = float(volatility)
        self.start_price = float(start_price)
        self.strike_price = float(strike_price)
        self.interest_rate = float(interest_rate)
        self.t_final = t_final
        self.true_measure = BrownianMotion(sampler, t_final)
        self.call_put = call_put
        if self.call_put not in ('call', 'put'):
            raise ParameterError("call_put must be either 'call' or 'put'")
        
        self.multilevel_dims = multilevel_dims
        if self.multilevel_dims is not None: # multi-level problem
            self.dim_fracs = np.array(
                [0] + [float(self.multilevel_dims[i])/float(self.multilevel_dims[i-1]) for i in range(1, len(self.multilevel_dims))],
                dtype=float
            )
            self.max_level = len(self.multilevel_dims) - 1
            self.leveltype = 'fixed-multi'
            self.parent = True
            self.parameters.append('multilevel_dims') # self.parameters += ['multilevel_dims']
        else: # single level problem
            self.dim_frac = dim_frac
            self.leveltype = 'single'
            self.parent = False
            self.parameters.append('dim_frac') # self.parameters += ['dim_frac']

        super(Option, self).__init__(dimension_indv=1, dimension_comb=1, parallel=False)

    def _get_discounted_payoffs(self):
        """ABSTRACT METHOD"""
        raise MethodImplementationError(self, '_get_discounted_payoffs')

    def _dimension_at_level(self, level):
        return self.d if self.multilevel_dims is None else self.multilevel_dims[level]