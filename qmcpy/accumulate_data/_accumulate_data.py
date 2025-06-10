from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError, MethodImplementationError, _univ_repr, DimensionError
import pickle
import os

class AccumulateData(object):
    """ Accumulated Data abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        """ Initialize data instance """
        prefix = 'A concrete implementation of AccumulateData must have '
        if not hasattr(self,'parameters'):
            self.parameters = []

    def update_data(self):
        """ ABSTRACT METHOD to update the accumulated data."""
        raise MethodImplementationError(self, 'update_data')

    def __repr__(self):
        string = _univ_repr(self, 'AccumulateData', self.parameters + ['time_integrate'])
        for qmc_obj in [self.stopping_crit, self.integrand, self.true_measure, self.discrete_distrib]:
            if qmc_obj:
                string += '\n'+str(qmc_obj)
        return string

    def save(self, filepath):
        """
        Save the AccumulateData object to disk using pickle.
        Args:
            filepath (str): Path to save the file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Load an AccumulateData object from disk.
        Args:
            filepath (str): Path to the saved file.
        Returns:
            AccumulateData: The loaded object.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)
