"""
Definition for abstract classes: DiscreteDistribution, Measure
"""
from abc import ABC, abstractmethod
from numpy import array,ndarray

from .. import univ_repr,MeasureCompatibilityError,DimensionError

class DiscreteDistribution(ABC):
    """ Specifies and generates the components of :math:`a_n \sum_{i=1}^n w_i \delta_{\mathbf{x}_i}(\cdot)` """

    def __init__(self, accepted_measures, true_distrib=None, distrib_data=None):
        """
        Construct a list of DiscreteDistributions

        Args:
            accepted_measures (list of strings): Measure objects compatible with the DiscreteDistribution
            true_distrib (Measure): Distribution's' from which we can generate sample's'
        """
        super().__init__()
        if true_distrib:
            self.true_distrib = true_distrib
            if type(self.true_distrib).__name__ not in accepted_measures:
                raise MeasureCompatibilityError(type(self).__name__+' only accepts measures:'+str(accepted_measures))
            self.distrib_list = [type(self)() for i in range(len(self.true_distrib))]
            # Distribute attributes to each distribution in the list
            for i in range(len(self)):
                self[i].true_distrib = self.true_distrib[i]
                self[i].distrib_data = distrib_data[i] if distrib_data else None

    @abstractmethod
    def gen_distrib(self, n, m):
        """
        Generate j nxm samples from the true-distribution
        
        Args:       
            n (int): Number of observations
            m (int): Number of dimensions
        
        Returns:
            nxm (numpy array) 
        """
        pass

    def __len__(self): return len(self.distrib_list)
    def __iter__(self):
        for distribObj in self.distrib_list:
            yield distribObj
    def __getitem__(self,i): return self.distrib_list[i]
    def __setitem__(self,i,val): self.distrib_list[i]=val
    def __repr__(self): return univ_repr(self,'distrib_list')


class Measure(ABC):
    """
    Specifies the components of a general measure used to define an
    integration problem or a sampling method
    """

    def __init__(self,dimension=None,**kwargs):
        """
        Construct a list of measures

        Args:
            dimension (list of ints): Dimensions to be dispersed among list of Measures
            **kwargs (dictionary): Accepts keyword arguments into dictionary. 
                                   Disperses dictionary values among list of Measures
        """
        self.dimension = dimension
        super().__init__()
        if dimension:
            # Type check dimension
            if type(self.dimension)==int: self.dimension = array([self.dimension])
            if all(type(i)==int and i>0 for i in self.dimension): self.dimension = array(self.dimension)
            else: raise DimensionError("dimension must be an numpy.ndarray (or list) of positive integers")
            # Type check values of keyword arguments
            for key,val in kwargs.items():
                # format each value into a list of values with length equal to the number of integrands
                if type(kwargs[key])!= list and type(kwargs[key])!= ndarray:
                    kwargs[key] = [kwargs[key]]
                if len(kwargs[key]) == 1 and len(self.dimension)!=1:
                    kwargs[key] = kwargs[key]*len(self.dimension)
                if len(kwargs[key]) != len(self.dimension):
                    raise MeasureDataError(key+" must be a numpy.ndarray (or list) of len(dimension)")
            self.measure_list = [type(self)() for i in range(len(self.dimension))]
            # Create list of measures with appropriote dimensions and keyword arguments
            for i in range(len(self)):
                self[i].dimension = self.dimension[i]
                for key,val in kwargs.items():
                    setattr(self[i],key,val[i])

    def __len__(self): return len(self.measure_list)
    def __iter__(self):
        for measureObj in self.measure_list:
            yield measureObj
    def __getitem__(self,i): return self.measure_list[i]
    def __setitem__(self,i,val): self.measure_list[i] = val
    def __repr__(self): return univ_repr(self,'measure_list')

