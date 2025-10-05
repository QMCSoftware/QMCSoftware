from ..util import _univ_repr
import pickle
import gzip
import os
from pathlib import Path

class Data(object):

    def __init__(self, parameters):
        self.parameters = parameters

    def save(self, filepath, compress=False, protocol=None):
        """
        Save the Data object to a file.
        
        Args:
            filepath (str or Path): Path where to save the file
            compress (bool): Whether to use gzip compression. If True, appends .gz to filename
            protocol (int): Pickle protocol version to use
        """
        filepath = Path(filepath)
        
        if compress:
            # Append .gz if not already present
            if not str(filepath).endswith('.gz'):
                filepath = Path(str(filepath) + '.gz')
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a copy of the object with only pickleable attributes
        data_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(self, attr_name)
                    # Test if the attribute is pickleable
                    pickle.dumps(attr_value)
                    data_dict[attr_name] = attr_value
                except (TypeError, AttributeError, pickle.PicklingError):
                    # Skip non-pickleable attributes
                    pass
        
        if compress:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data_dict, f, protocol=protocol)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data_dict, f, protocol=protocol)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a Data object from a file.
        
        Args:
            filepath (str or Path): Path to the file to load
            
        Returns:
            Data: Loaded Data object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"No such file or directory: '{filepath}'")
        
        # Try to detect if file is compressed by checking if it can be opened with gzip
        try:
            with gzip.open(filepath, 'rb') as f:
                data_dict = pickle.load(f)
        except (gzip.BadGzipFile, OSError):
            # Not a gzip file, try regular pickle
            with open(filepath, 'rb') as f:
                data_dict = pickle.load(f)
        
        # Create a new Data object and restore the attributes
        if isinstance(data_dict, dict):
            # New format - dictionary of attributes
            parameters = data_dict.get('parameters', [])
            data_obj = cls(parameters)
            for attr_name, attr_value in data_dict.items():
                if attr_name != 'parameters':  # parameters already set in __init__
                    setattr(data_obj, attr_name, attr_value)
            return data_obj
        else:
            # Old format - direct object (fallback)
            return data_dict

    def __repr__(self):
        string = _univ_repr(self, 'Data', self.parameters + ['time_integrate'])
        if hasattr(self,"stopping_crit") and self.stopping_crit:
            string += '\n'+str(self.stopping_crit)
        if hasattr(self,"integrand") and self.integrand:
            string += '\n'+str(self.integrand)
        if hasattr(self,"true_measure") and self.true_measure:
            string += '\n'+str(self.true_measure)
        if hasattr(self,"discrete_distrib") and self.discrete_distrib:
            string += '\n'+str(self.discrete_distrib)
        return string
