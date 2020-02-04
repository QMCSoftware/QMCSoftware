""" Distribution is an abstract class. """

from ..util import ParameterError, MethodImplementationError, univ_repr
from numpy import array


class Distribution(object):
    """ Discrete Distribution from which we can generate samples """

    def __init__(self):
        prefix = 'A concrete implementation of Distribution must have '
        if not hasattr(self, 'mimics'):
            raise ParameterError(prefix + 'self.mimcs (measure mimiced by the distribution)')
        if not hasattr(self, 'dimension'):
            raise ParameterError(prefix + 'self.dimension')
    
    def gen_samples(self, *args):
        """ ABSTRACT METHOD
        Generate self.replications (n_max-n_min)xself.d Lattice samples
        
        Returns:
            self.replications x (n_max-n_min) x self.dimension (ndarray)
        """
        raise MethodImplementationError(self, 'gen_dd_samples')

    def __repr__(self, attributes=[]):
        """
        Print important attribute values

        Args:
            attributes (list): list of attributes to print

        Returns:
            string of self info
        """
        return univ_repr(self, "Discrete Distribution", attributes)
