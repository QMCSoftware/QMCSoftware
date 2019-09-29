''' Originally developed in MATLAB by Fred Hickernell. Translated to python
by Sou-Cheng T. Choi and Aleksei Sorokin '''
from abc import ABC, abstractmethod
from math import inf, nan

from algorithms.distribution import discreteDistribution
from algorithms.function import fun
from numpy import array

from .. import univ_repr


class AccumData(ABC):
    '''
    Accumulated data required in the computation of the integral
        Any sublcass of AccumData must include:
            Methods: updateData(self, distribObj, fun_obj, decompType)
    '''

    def __init__(self):
        super().__init__()
        self.solution = nan  # solution
        self.stage = 'begin'  # stage of computation; is 'done' when finished
        self.prevN = array(
            [])  # new data will be based on (quasi-)random vectors indexed by
        self.nextN = array([])  # prevN + 1 to nextN
        self.timeUsed = array([])  # time used so far
        self.nSamplesUsed = array([])  # number of samples used so far
        self.confidInt = array([-inf, inf])  # error bound on the solution
        self.costF = array([])  # time required to compute function values
        self._timeStart = None  # hidden/private

    # Abstract Methods
    @abstractmethod
    def updateData(self, distrib_obj: discreteDistribution, fun_obj: fun,
                   decomp_type):
        """
        Update the accumulated data

        Args:
            distrib_obj: an instance of discreteDistribution
            fun_obj: an instance of function
            decomp_type:

        Returns:
            None

        """
        pass

    # Magic Method
    def __repr__(self): return univ_repr(self)
