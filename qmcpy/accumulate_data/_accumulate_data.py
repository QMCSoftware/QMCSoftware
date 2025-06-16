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

    def save(self, filepath, protocol=None, compress=False):
        """
        Save the AccumulateData object to disk using optimized pickle.
        
        Args:
            filepath (str): Path to save the file.
            protocol (int, optional): Pickle protocol version. If None, uses highest available.
                                    Higher protocols are generally faster and produce smaller files.
            compress (bool): Whether to use compression. Can significantly reduce file size
                           for large numpy arrays at the cost of some speed.
        
        Performance Notes:
            - protocol=pickle.HIGHEST_PROTOCOL is fastest (currently 5 in Python 3.8+)
            - protocol=5 supports out-of-band data transfer for large numpy arrays
            - compress=True uses gzip compression (good for large arrays with repetitive data)
            - When compress=True, '.gz' is automatically appended to filename if not present
        """
        if protocol is None:
            protocol = pickle.HIGHEST_PROTOCOL
            
        if compress:
            import gzip
            # Auto-append .gz extension if not already present
            filepath = str(filepath)
            if not (filepath.endswith('.gz') or filepath.endswith('.gzip')):
                filepath = filepath + '.gz'
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=protocol)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=protocol)

    @staticmethod
    def load(filepath, compressed=None):
        """
        Load an AccumulateData object from disk.
        
        Args:
            filepath (str): Path to the saved file.
            compressed (bool, optional): Whether file is compressed. If None, auto-detects.
        
        Returns:
            AccumulateData: The loaded object.
        """
        # Convert Path objects to string
        filepath = str(filepath)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
            
        # Auto-detect compression if not specified
        if compressed is None:
            compressed = filepath.endswith('.gz') or filepath.endswith('.gzip')
            
        if compressed:
            import gzip
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                return pickle.load(f)

    