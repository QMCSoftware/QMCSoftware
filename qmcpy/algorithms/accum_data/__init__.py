from abc import ABC, abstractmethod
from math import inf, nan
from numpy import array

from algorithms.distribution import DiscreteDistribution
from algorithms.integrand import Integrand
from .. import univ_repr

class AccumData(ABC):
    """ Accumulated data required in the computation of the integral """

    def __init__(self):
        super().__init__()
        self.solution = nan  # solution
        self.stage = 'begin'  # stage of computation; is 'done' when finished
        self.prevN = array(
            [])  # new data will be based on (quasi-)random vectors indexed by
        self.nextN = array([])  # prevN + 1 to nextN
        self.timeUsed = array([])  # time used so far
        self.nSamplesUsed = array([])  # number of samples used so far
        self.confidInt = array([-inf, inf])  # error bound on the solution
        self.cost_eval = array([])  # time required to compute integrand values
        self._timeStart = None  # hidden/private

    @abstractmethod
    def update_data(self, distrib_obj: DiscreteDistribution, fun_obj: Integrand, decomp_type):
        """
        Update the accumulated data

        Args:
            distrib_obj: an instance of DiscreteDistribution
            fun_obj: an instance of Integrand
            decomp_type:

        Returns:
            None

        """
        pass

    def __repr__(self): return univ_repr(self)
