""" Definition for abstract class AccumData """

from ..util import ParameterError, MethodImplementationError, univ_repr


class AccumData(object):
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
        prefix = 'A concrete implementation of AccumData must have '
        if not hasattr(self, 'solution'):
            raise ParameterError(prefix + 'self.solution')
        if not hasattr(self, 'n'):
            raise ParameterError(prefix + 'self.n (current number of samples at each level)')
        if not hasattr(self, 'n_total'):
            raise ParameterError(prefix + 'self.n_total (total number of samples)')
        if not hasattr(self, 'confid_int'):
            raise ParameterError(prefix + 'self.confid_int (confidence interval for the solution)')

    def update_data(self, integrand, measure):
        """
        ABSTRACT METHOD
        Update the accumulated data

        Args:
            integrand (Integrand): an instance of Integrand
            measure (Measure): an instance of Measure

        Returns:
            None
        """
        raise MethodImplementationError(self, 'update_data')

    def complete(self, time_total, integrand=None, distrib=None,
                 measure=None, stopping_criterion=None):
        """
        Aggregate all objects after integration completes

        Args:
            time_total (float): total wall clock time for integration
            integrand (Integrand): Integrand object
            distrib (Distribution): Discrete Distribution object
            measure (Measure): True Measure Object
            stopping_criterion (Stopping Criterion): Stopping Criterion object

        Returns:
            self
        """
        self.time_total = time_total
        self.integrand = integrand
        self.distrib = distrib
        self.measure = measure
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
        for qmc_obj in [self.integrand, self.distrib,
                        self.measure, self.stopping_criterion]:
            if qmc_obj:
                string += str(qmc_obj)
        super_attributes = ['n', 'n_total', 'confid_int', 'time_total']
        #   get only unique values
        string += univ_repr(self, 'AccumData', super_attributes + attributes)
        return string
