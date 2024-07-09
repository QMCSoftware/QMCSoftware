from tracemalloc import start
from qmcpy import *
import numpy as np
import scipy as sc
from ._integrand import Integrand
from ..true_measure import BrownianMotion
from ..util import ParameterError

class Option(Integrand):
    """Option abstract class. DO NOT INSTANTIATE."""

    def __init__(self, volatility:float, start_price:float, strik_price:float, interest_rate:float, t_final:float, call_put:str):
        """
        Args:
            volatility (float): sigma, the volatility of the asset
            start_price (float): S(0), the asset value at t = 0
            strike_price (float): the call/put offer
            interest_rate (float): r, the annual interest rate
            t_final (float): exercise time
            call_put (str): 'call' or 'put' option
        """
        self.paramters = ['volatility', 'call_put', 'start_price', 'strike_price', 'interest_rate']
        self.volatility = volatility
        self.start_price = start_price
        self.strike_price = strik_price
        self.interest_rate = interest_rate
        self.t_final = t_final
        self.call_put = call_put
        if self.call_put not in ('call', 'put'):
            raise ParameterError("call_put must be either 'call' or 'put'")
        super(Option, self).__init__(dimension_indv=1, dimension_comb=1, parallel=False)