from numpy import array,ndarray,int64
from abc import ABC


from qmcpy._util import DimensionError,TransformError,univ_repr

class TrueDistribution(ABC):
    """ Samples from the Discrete Distribution will be transformed into the True Distribution"""

    def __init__(self, dimension, transforms, **kwargs):
        self.dimension = dimension
        super().__init__()
        if not dimension:
            return
        # Type check dimension
        if type(self.dimension) == int:
            self.dimension = array([self.dimension])
        if all(type(i)==int or type(i)==int64 and i > 0 for i in self.dimension):
            self.dimension = array(self.dimension)
        else:
            msg = "dimension must be an numpy.ndarray/list of positive integers"
            raise DimensionError(msg)
        # Type check measureData
        for key, val in kwargs.items():
            if type(kwargs[key]) != list and type(kwargs[key]) != ndarray:
                kwargs[key] = [kwargs[key]]
            if len(kwargs[key]) == 1 and len(self.dimension) != 1:
                kwargs[key] = kwargs[key] * len(self.dimension)
            if len(kwargs[key]) != len(self.dimension):
                msg =  " must be a numpy.ndarray (or list) of len(dimension)"
                raise DimensionError(key + msg)
        self.distributions = [type(self)(None) for i in range(len(self.dimension))]
        # Create list of measures with proper dimensions and keyword arguments
        for i in range(len(self)):
            self[i].dimension = self.dimension[i]
            for key, val in kwargs.items():
                setattr(self[i], key, val[i])
            self[i].transforms = transforms
    
    def transform_generator(self,discrete_distribution):
        for i in range(len(self)):
            try: # Try to wrap the distribution   
                self[i].gen_distribution = lambda n_streams,n_obs,self=self[i]:\
                    self.transforms[discrete_distribution.mimics](self,\
                        discrete_distribution.gen_samples(int(n_streams),int(n_obs),int(self.dimension))) 
            except: raise TransformError('Cannot tranform %s to %s'\
                %(type(discrete_distribution).__name__,type(self).__name__))

    def __len__(self):
        return len(self.distributions)

    def __iter__(self):
        for distribution in self.distributions:
            yield distribution

    def __getitem__(self, i):
        return self.distributions[i]

    def __setitem__(self, i, val):
        self.distributions[i] = val

    def __repr__(self):
        return univ_repr(self, "True Distribution")

    
    def summarize(self):
        """Print important attribute values
        """
        header_fmt = "%s (%s)"
        item_s = "%35s: %-15s"
        attrs_vals_str = ""

        attrs_vals_str += header_fmt % (type(self).__name__, "True Distribution Object")
        print(attrs_vals_str)
    

# API
from .true_distributions import *