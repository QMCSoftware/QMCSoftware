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
        self.n_prev = array([])  # new data will be based on (quasi-)random vectors indexed by
        self.n_next = array([])  # n_prev + 1 to n_next
        self.n_samples_total = array([])  # number of samples used so far
        self.confidInt = array([-inf, inf])  # error bound on the solution
        self.t_total = None  # total computation time. Set by integrate method

    @abstractmethod
    def update_data(self, distrib_obj: DiscreteDistribution, fun_obj: Integrand):
        """
        Update the accumulated data

        Args:
            distrib_obj: an instance of DiscreteDistribution
            fun_obj: an instance of Integrand

        Returns:
            None

        """
        pass

    def __repr__(self): return univ_repr(self)
