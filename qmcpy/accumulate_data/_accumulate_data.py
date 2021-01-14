from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError, MethodImplementationError, _univ_repr, DimensionError

class AccumulateData(object):
    """ Accumulated Data abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        """ Initialize data instance """
        prefix = 'A concrete implementation of AccumulateData must have '
        if not hasattr(self,'stopping_crit'):
            raise ParameterError(prefix + 'self.stopping_crit (a StoppingCriterion)')
        if not hasattr(self,'integrand'):
            raise ParameterError(prefix + 'self.integrand (an Integrand)')
        if not hasattr(self,'true_measure'):
            raise ParameterError(prefix + 'self.true_measure (a TrueMeasure)')
        if not hasattr(self,'discrete_distrib'):
            raise ParameterError(prefix + 'self.discrete_distrib (a DiscreteDistribution)')
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
        for qmc_obj in [self.integrand, self.discrete_distrib, self.true_measure, self.stopping_crit]:
            if qmc_obj:
                string += str(qmc_obj)+'\n'
        string += _univ_repr(self, 'AccumulateData', self.parameters + ['time_integrate'])
        return string
