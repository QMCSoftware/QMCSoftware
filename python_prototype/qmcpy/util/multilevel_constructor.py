""" Multi-level abstract class """

from ..integrand._integrand import Integrand
from ..true_measure._true_measure import TrueMeasure
from . import univ_repr
from numpy import repeat


class MultiLevelConstructor(object):
    """ Abstract Multi-Level Class """

    def __init__(self, levels, qmcpy_object, **kwargs):
        """
        Args:
            levels (int): number of levels
            qmcpy_object (object): object to construct from distributing kwargs
            kwargs (dict): keyword arguments to distribute when constructing qmcpy_object
                key (str): keyword
                val (list/ndarray): list of length levels.
                                    val[i] will go toward constructing qmcpy_object[i]
                                    Will also accept single arguments and repeat levels times
        """
        self.kwargs = kwargs
        self.ex_qmcpy_obj = object.__new__(qmcpy_object) 
        if isinstance(self.ex_qmcpy_obj,Integrand):
            self.measure = kwargs['measure']
        elif isinstance(self.ex_qmcpy_obj,TrueMeasure):
            self.distribution = kwargs['distribution']
        self.name = type(self.ex_qmcpy_obj).__name__
        self.levels = levels
        for key,val in self.kwargs.items():
            if not hasattr(val,'__len__') or len(val)!=self.levels:
                self.kwargs[key] = repeat(val,self.levels)
        if hasattr(qmcpy_object,'add_multilevel_kwargs'):
            self.kwargs = qmcpy_object.add_multilevel_kwargs(self.levels,**self.kwargs)
        self.objects_list = [None]*self.levels
        for l in range(self.levels):
            kwargs_l = {key:val[l] for key,val in self.kwargs.items()}
            self.objects_list[l] = qmcpy_object(**kwargs_l)
        self.dimensions = [obj.dimension for obj in self.objects_list]
        for param in self.ex_qmcpy_obj.parameters:
            setattr(self,param,[getattr(obj,param) for obj in self.objects_list])        
    
    def __len__(self):
        return len(self.objects_list)

    def __iter__(self):
        for obj in self.objects_list:
            yield obj

    def __getitem__(self, i):
        return self.objects_list[i]

    def __setitem__(self, i, val):
        self.objects_list[i] = val
    
    def __repr__(self):
        return univ_repr(self,self.name,self.ex_qmcpy_obj.parameters)