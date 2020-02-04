""" Multi-level abstract class """

from .util import univ_repr
from numpy import repeat


class MultiLevelConstructor():
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
        self.ex_qmcpy_obj = object.__new__(qmcpy_object) 
        self.name = type(self.ex_qmcpy_obj).__name__
        self.levels = levels
        for key,val in kwargs.items():
            setattr(self,key,val)
        self.kwargs = kwargs
        for key,val in self.kwargs.items():
            if not hasattr(val,'__len__') or len(val)!=self.levels:
                self.kwargs[key] = repeat(val,self.levels)
        if hasattr(qmcpy_object,'add_multilevel_kwargs'):
            self.kwargs = qmcpy_object.add_multilevel_kwargs(self.levels,**self.kwargs)
        self.objects_list = [None]*self.levels
        for l in range(self.levels):
            kwargs_l = {key:val[l] for key,val in self.kwargs.items()}
            self.objects_list[l] = qmcpy_object(**kwargs_l)
    
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
        name = self.name+' (MultiLevel)'
        return univ_repr(self,name,self.ex_qmcpy_obj.parameters)