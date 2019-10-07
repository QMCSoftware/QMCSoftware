""" Definition for abstract class AccumData """

from abc import ABC, abstractmethod
from math import inf, nan
from numpy import array

from .._util import univ_repr

class AccumData(ABC):
    """
    Accumulated data required in the computation of the integral, stores the sample mean and variance of integrand values

    Attributes:
        stage (str): stage of computation; 'begin', or 'done' when finished
        n_samples_total (array-like): number of samples used so far
        confid_int (array-like (2, 1)): error bound on the solution
        t_total (float): total computation time. Set by integrate method.
    """

    def __init__(self):
        """ Initialize data instance """
        super().__init__()
        self.solution = nan # solution
        self.stage = 'begin'
        self.n_prev = array([])  # new data will be based on (quasi-)random vectors indexed by.
        self.n_next = array([])  # n_prev + 1 to n_next.
        self.n_samples_total = array([])
        self.confid_int = array([-inf, inf])
        self.t_total = None

    @abstractmethod
    def update_data(self, distribution, integrand):
        """
        Update the accumulated data

        Args:
            distribution (DiscreteDistribution): an instance of DiscreteDistribution
            integrand (Integrand): an instance of Integrand
        Returns:
            None
        """

    def __repr__(self):
        return univ_repr(self)

    def summarize(self):
        h1 = '%s (%s)\n'
        item_f = '%25s: %-15.4f\n'
        item_s = '%25s: %-15s\n'

        s = 'Solution: %-15.4f\n%s' % (self.solution, '~' * 50)
        print(s)

        if not self.integrand is None: self.integrand.summarize()
        if not self.measure is None: self.measure.summarize()
        if not self.distribution is None: self.distribution.summarize()
        if not self.stopping_criterion is None: self.stopping_criterion.summarize()

        s = h1 % (type(self).__name__, 'Data Object')
        s += item_s % ('n_samples_total', str(self.n_samples_total))
        s += item_f % ('t_total', self.t_total)
        s += item_s % ('confid_int', str(self.confid_int))
        print(s[:-1])


# API
from .mean_var_data import MeanVarData
from .mean_var_data_rep import MeanVarDataRep