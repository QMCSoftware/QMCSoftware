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
        stage (str): stage of computation; "begin", or "done" when finished
        n_samples_total (array-like): number of samples used so far
        confid_int (array-like (2, 1)): error bound on the solution
        time_total (float): total computation time. Set by integrate method.
    """

    def __init__(self):
        """ Initialize data instance """
        super().__init__()
        self.solution = nan  # solution
        self.stage = "begin"
        # new data will be based on (quasi-)random vectors indexed by.
        self.n = array([])  # number of samples at this stage
        self.n_total = 0
        self.confid_int = array([-inf, inf])
        self.time_total = None
        self.integrand = None
        self.discrete_distrib = None
        self.true_measure = None
        self.stopping_criterion = None

    @abstractmethod
    def update_data(self, integrand, true_measure):
        """
        Update the accumulated data

        Args:
            integrand (Integrand): an instance of Integrand
            true_measure (TrueMeasure): an instance of TrueMeasure

        Returns:
            None
        """
    
    def complete(self, time_total, integrand=None, discrete_distrib=None, true_measure=None, stopping_criterion=None):
        """
        Aggregate all objects after integration completes
        
        Args: 
            time_total (float): total wall clock time for integration
            integrand (Integrand): Integrand object
            discrete_distrib (DiscreteDistribution): Discrete Distribution object
            true_measure (TrueMeasure): True Measure Object
            stopping_criterion (Stopping Criterion): Stopping Criterion object
        
        Returns: 
            self
        """
        self.time_total = time_total
        self.integrand = integrand
        self.discrete_distrib = discrete_distrib
        self.true_measure = true_measure
        self.stopping_criterion = stopping_criterion
        return self

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args: 
            attributes (list): list of attributes to print
        
        Returns:
            string of self info
        """
        string = "Solution: %-15.4f\n" % (self.solution)
        for qmc_obj in [self.integrand, self.discrete_distrib, \
                        self.true_measure, self.stopping_criterion]:
            if qmc_obj:
                string += str(qmc_obj)
        data_attributes = set(attributes + ['n', 'n_total', 'confid_int', 'time_total'])
        #   get only unique values
        string += univ_repr(self, 'AccumData', data_attributes)
        return string
