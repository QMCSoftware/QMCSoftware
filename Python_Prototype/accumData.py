from abc import ABC, abstractmethod
from math import inf, nan


# Accumulated data required in the computation of the integral
class accumData(ABC):

    def __init__(self):
        self.solution = nan  # solution
        self.stage = 'begin'  # stage of the computation, becomes 'done' when finished
        self.prevN  # new data will be based on (quasi-)random vectors indexed by
        self.nextN  # prevN + 1 to nextN
        self.timeUsed  # time used so far
        self.nSamplesUsed  # number of samples used so far
        self.errorBound = [-inf, inf]  # error bound on the solution
        self.costF  # time required to compute function values

    # Abstract Properties
    @property
    @abstractmethod
    def timeStart(self):  # %starting time
        pass

    # Abstract Method
    @abstractmethod
    def updateData(self, distribObj, fun_obj, decompType):  # update the accumulated data
        pass
