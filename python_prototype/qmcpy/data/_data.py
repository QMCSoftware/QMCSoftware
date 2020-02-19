""" Definition for abstract class Data """

from ..util import ParameterError, MethodImplementationError, univ_repr


class Data():
    """
    Accumulated data required in the computation of the integral.
    """

    def __init__(self):
        """ Initialize data instance """
        prefix = 'A concrete implementation of Data must have '
        if not hasattr(self, 'solution'):
            raise ParameterError(prefix + 'self.solution')
        if not hasattr(self, 'n_total'):
            raise ParameterError(prefix + 'self.n_total (total number of samples)')
        if not hasattr(self,'parameters'):
            self.parameters = []
        self.time_total = -1

    def update_data(self, integrand, measure):
        """ ABSTRACT METHOD
        Update the accumulated data

        Args:
            integrand (Integrand): an instance of Integrand
            measure (Measure): an instance of Measure

        Returns:
            None
        """
        raise MethodImplementationError(self, 'update_data')

    def complete(self, time_total, integrand, distribution, measure, stopping_criterion):
        """
        Aggregate all objects after integration completes

        Args:
            time_total (float): total wall clock time for integration
            integrand (Integrand): Integrand object
            distribution (Distribution): Discrete Distribution object
            measure (Measure): True Measure Object
            stopping_criterion (Stopping Criterion): Stopping Criterion object

        Returns:
            self
        """
        self.time_total = time_total
        self.integrand = integrand
        self.distribution = distribution
        self.measure = measure
        self.stopping_criterion = stopping_criterion

    def __repr__(self):
        string = "Solution: %-15.4f\n" % (self.solution)
        for qmc_obj in [self.integrand, self.distribution, self.measure, self.stopping_criterion]:
            if qmc_obj:
                string += str(qmc_obj)
        string += univ_repr(self, 'Data', self.parameters + ['time_total'])
        return string
