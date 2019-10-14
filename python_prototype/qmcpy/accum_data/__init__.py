""" Definition for abstract class AccumData """

from abc import ABC, abstractmethod
from math import inf, nan

from numpy import array
from .._util import univ_repr


class AccumData(ABC):
    """
    Accumulated data required in the computation of the integral, stores the \
    sample mean and variance of integrand values

    Attributes:
        stage (str): stage of computation; 'begin', or 'done' when finished
        n_samples_total (array-like): number of samples used so far
        confid_int (array-like (2, 1)): error bound on the solution
        t_total (float): total computation time. Set by integrate method.
    """

    def __init__(self):
        """ Initialize data instance """
        super().__init__()
        self.solution = nan  # solution
        self.stage = "begin"
        self.n_prev = array([])
        # new data will be based on (quasi-)random vectors indexed by.
        self.n_next = array([])  # n_prev + 1 to n_next.
        self.n_samples_total = array([])
        self.confid_int = array([-inf, inf])
        self.t_total = None
        self.integrand = None
        self.discrete_distrib = None
        self.true_measure = None
        self.stopping_criterion = None

    @abstractmethod
    def update_data(self, true_measure, integrand):
        """
        Update the accumulated data

        Args:
            true_measure (TrueMeasure): an instance of TrueMeasure
            integrand (Integrand): an instance of Integrand

        Returns:
            None
        """

    def __repr__(self):
        return univ_repr(self)

    def summarize(self):
        """Print important attribute values
        """
        header_fmt = "%s (%s)\n"
        item_f = "%25s: %-15.4f\n"
        item_s = "%25s: %-15s\n"

        attrs_vals_str = "Solution: %-15.4f\n%s" % (self.solution, "~" * 50)
        print(attrs_vals_str)

        if self.integrand:
            self.integrand.summarize()
        if self.discrete_distrib:
            self.discrete_distrib.summarize()
        if self.true_measure:
            self.true_measure.summarize()
        if self.stopping_criterion:
            self.stopping_criterion.summarize()

        attrs_vals_str = header_fmt % (type(self).__name__, "Data Object")
        attrs_vals_str += item_s % ("n_samples_total",
                                    str(self.n_samples_total))
        attrs_vals_str += item_f % ("t_total", self.t_total)
        attrs_vals_str += item_s % ("confid_int", str(self.confid_int))
        print(attrs_vals_str[:-1]+'\n')

# API
from .mean_var_data import MeanVarData
from .mean_var_data_rep import MeanVarDataRep