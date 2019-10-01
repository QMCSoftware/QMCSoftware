from numpy import arange

from . import Measure
    
class StdUniform(Measure):
    def __init__(self,dimension=None):super().__init__(dimension)

class StdGaussian(Measure):
    def __init__(self,dimension=None):super().__init__(dimension)

class IIDZeroMeanGaussian(Measure):
    def __init__(self,dimension=None,variance=None):
        super().__init__(dimension,variance=variance)

class BrownianMotion(Measure):
    def __init__(self,timeVector=None):
        if timeVector: dimension = [len(tV) for tV in timeVector]
        else: dimension = None
        super().__init__(dimension,timeVector=timeVector)

class Lattice(Measure):
    def __init__(self,dimension=None):
        super().__init__(dimension,mimics='StdUniform')

class Sobol(Measure):
    def __init__(self,dimension=None):
        super().__init__(dimension,mimics='StdUniform')   