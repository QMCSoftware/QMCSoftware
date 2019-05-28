from abc import ABC, abstractmethod
from math import inf, nan
from numpy import array
from util import univ_repr

# Accumulated data required in the computation of the integral
class accumData(ABC):
    '''
    Any sublcass of accumData must include:
        Properties: timeStart
        Methods: updateData(self, distribObj, fun_obj, decompType)
    '''
    def __init__(self):
        super().__init__()
        self.solution = nan  # solution
        self.stage = 'begin'  # stage of the computation, becomes 'done' when finished
        self.prevN = array([])  # new data will be based on (quasi-)random vectors indexed by
        self.nextN = array([])  # prevN + 1 to nextN
        self.timeUsed = array([])  # time used so far
        self.nSamplesUsed = array([])  # number of samples used so far
        self.confidInt = array([-inf, inf])  # error bound on the solution
        self.costF = array([])  # time required to compute function values
        self._timeStart = None

    # Abstract Methods
    @abstractmethod
    def updateData(self, distribObj, fun_obj, decompType):  # update the accumulated data
        pass
    
    def __repr__(self): return univ_repr(self,'accumData')
        
