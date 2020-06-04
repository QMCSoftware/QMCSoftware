from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError, MethodImplementationError, univ_repr, DimensionError

class AccumulateData():
    """ Accumulated Data abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        """ Initialize data instance """
        prefix = 'A concrete implementation of AccumulateData must have '
        if not hasattr(self,'stopping_criterion'):
            raise ParameterError(prefix + 'self.stopping_criterion (a StoppingCriterion)')
        if not hasattr(self,'integrand'):
            raise ParameterError(prefix + 'self.integrand (an Integrand)')
        if not hasattr(self,'measure'):
            raise ParameterError(prefix + 'self.measure (a TrueMeasure)')
        if not hasattr(self,'distribution'):
            raise ParameterError(prefix + 'self.distribution (a DiscreteDistribution)')
        if not hasattr(self, 'solution'):
            raise ParameterError(prefix + 'self.solution')
        if not hasattr(self, 'n_total'):
            raise ParameterError(prefix + 'self.n_total (total number of samples)')
        if not hasattr(self,'parameters'):
            self.parameters = []

    def update_data(self):
        """ ABSTRACT METHOD to update the accumulated data."""
        raise MethodImplementationError(self, 'update_data')

    def __repr__(self):
        string = "Solution: %-15.4f\n" % (self.solution)
        for qmc_obj in [self.integrand, self.distribution, self.measure, self.stopping_criterion]:
            if qmc_obj:
                string += str(qmc_obj)
        string += univ_repr(self, 'AccumulateData', self.parameters + ['time_integrate'])
        return string
