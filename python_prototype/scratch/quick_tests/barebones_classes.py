""" Sample Barebones Subclasses """

from qmcpy.accum_data._accum_data import AccumData
from qmcpy.distribution._distribution import Distribution
from qmcpy.integrand._integrand import Integrand
from qmcpy.stopping_criterion._stopping_criterion import StoppingCriterion
from qmcpy.measure._measure import Measure

from numpy import *


# Accumulate Data
class MyAccumData(AccumData):
    
    def __init__(self):
        self.solution = None
        self.n = None
        self.n_total = None
        self.confid_int = None
        super().__init__()
    
    def update_data(self, integrand, measure):
        return

my_accum_data =  MyAccumData()
my_accum_data.update_data(None, None)


# Discrete Distribution
class MyDiscreteDistribution(Distribution):
    
    def __init__(self):
        self.mimics = None
        super().__init__()
    
    def gen_dd_samples(self, replications, n_samples, dimensions):
        return zeros((replications, n_samples, dimensions))

my_discrete_distribution = MyDiscreteDistribution()
samples = my_discrete_distribution.gen_dd_samples(1, 2, 3)


# Integrand
class MyIntegrand(Integrand):

    def __init__(self, dimensions):
        super().__init__(dimensions)
    
    def g(self,x):
        return x.sum(1)

my_integrand = MyIntegrand([2])
evalutations = my_integrand.g(ones((3,5)))


# Stopping Criterion
class MyStoppingCriterion(StoppingCriterion):

    def __init__(self, distribution):
        self.abs_tol = None
        self.rel_tol = None
        self.n_max = None
        self.stage = None
        super().__init__(distribution, ['MyDiscreteDistribution'])
    
    def stop_yet(self):
        return None

my_stopping_criterion = MyStoppingCriterion(MyDiscreteDistribution())
my_stopping_criterion.stop_yet()

# True Measure
class MyTrueMeasure(Measure):

    def __init__(self, dimension):
        transforms = {
            "StdMeasure": [
                lambda self, samples: samples,
                lambda self, g: g]}
        super().__init__(dimension, None)

my_true_measure = MyTrueMeasure([2])