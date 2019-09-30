''' Originally developed in MATLAB by Fred Hickernell. Translated to python by Sou-Cheng T. Choi and Aleksei Sorokin '''
from abc import ABC, abstractmethod
from numpy import array,ndarray

from .. import univ_repr

class MeasureCompatibilityError(Exception): pass
class DimensionError: pass

class DiscreteDistribution(ABC):
    '''
    Specifies and generates the components of :math:`a_n \sum_{i=1}^n w_i \delta_{\mathbf{x}_i}(\cdot)`
        Any sublcass of DiscreteDistribution must include:
            Methods: gen_distrib(self,nStart,nEnd,n,coordIndex)
            Properties: distrib_data,trueD
    '''

    def __init__(self, accepted_measures, trueD=None, distrib_data=None):
        super().__init__()  
        if trueD:
            self.trueD = trueD
            if type(self.trueD).__name__ not in accepted_measures:
                raise MeasureCompatibilityError(type(self).__name__+' only accepts measures:'+str(accepted_measures))
            self.distrib_list = [type(self)() for i in range(len(self.trueD))]
            for i in range(len(self)):    
                self[i].trueD = self.trueD[i]
                self[i].distrib_data = distrib_data[i] if distrib_data else None

    # Abstract Methods
    @abstractmethod
    def gen_distrib(self, n, m, j):
        """
         nStart = starting value of :math:`i`

         nEnd = ending value of :math:`i`

         n = value of :math:`n` used to determine :math:`a_n`

         coordIndex = which coordinates in sequence are needed
        """
        pass
    
    # Magic Methods. Makes self[i]==self.distrib_list[i]
    def __len__(self): return len(self.distrib_list)
    def __iter__(self):
        for distribObj in self.distrib_list:
            yield distribObj
    def __getitem__(self,i): return self.distrib_list[i]
    def __setitem__(self,i,val): self.distrib_list[i]=val
    def __repr__(self): return univ_repr(self,'distrib_list')


class Measure(ABC):
    '''
    Specifies the components of a general Measure used to define an
    integration problem or a sampling method
    '''
    
    def __init__(self,dimension=None,**kwargs):  
        self.dimension = dimension
        super().__init__()
        if dimension:
            # Type check dimension
            if type(self.dimension)==int: self.dimension = array([self.dimension])
            if all(type(i)==int and i>0 for i in self.dimension): self.dimension = array(self.dimension)
            else: raise DimensionError("dimension must be an numpy.ndarray (or list) of positive integers")
            # Type check measureData
            for key,val in kwargs.items():
                if type(kwargs[key])!= list and type(kwargs[key])!= ndarray:
                    kwargs[key] = [kwargs[key]]
                if len(kwargs[key]) == 1 and len(self.dimension)!=1:
                    kwargs[key] = kwargs[key]*len(self.dimension)
                if len(kwargs[key]) != len(self.dimension):
                    raise MeasureDataError(key+" must be a numpy.ndarray (or list) of len(dimension)")
            self.measure_list = [type(self)() for i in range(len(self.dimension))]
            for i in range(len(self)):    
                self[i].dimension = self.dimension[i]
                for key,val in kwargs.items():
                    setattr(self[i],key,val[i])

    # Magic Methods. Makes self[i]==self.measure_list[i]
    def __len__(self): return len(self.measure_list)
    def __iter__(self):
        for measureObj in self.measure_list:
            yield measureObj
    def __getitem__(self,i): return self.measure_list[i]
    def __setitem__(self,i,val): self.measure_list[i] = val
    def __repr__(self): return univ_repr(self,'measure_list')

