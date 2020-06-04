""" Sample Barebones Subclasses """

from qmcpy.accumulate_data._accumulate_data import AccumulateData
from qmcpy.discrete_distribution._discrete_distribution import DiscreteDistribution
from qmcpy.integrand._integrand import Integrand
from qmcpy.stopping_criterion._stopping_criterion import StoppingCriterion
from qmcpy.true_measure._true_measure import TrueMeasure
from qmcpy.util import TransformError
from numpy import *


def barebones():
        
    # Discrete DiscreteDistribution
    class MyDiscreteDistribution(DiscreteDistribution):
        
        def __init__(self, dimension):
            self.dimension = dimension
            self.mimics = 'StdMeasure'
            super().__init__()
        
        def gen_samples(self, n):
            return zeros((self.dimension,n))

    my_discrete_distribution = MyDiscreteDistribution(2)
    samples = my_discrete_distribution.gen_samples(4)


    # True TrueMeasure
    class MyTrueMeasure(TrueMeasure):

        def __init__(self, distribution):
            self.distribution = distribution
            super().__init__()
        
        def gen_samples(*args):
            samples = self.distribution.gen_samples(*args)
            if self.distribution.mimics == 'StdMeasure':
                tf_samples = samples
            else:
                raise TransformError('distribution must mimic StdMeasure')
            return tf_samples
        
        def transform_g_to_f(self,g):
            f = lambda samples: g(samples)
            return f

    my_true_measure = MyTrueMeasure(my_discrete_distribution)


    # Integrand
    class MyIntegrand(Integrand):

        def __init__(self, measure):
            self.measure = measure
            super().__init__()
        
        def g(self,x):
            return x.sum(1)

    my_integrand = MyIntegrand(my_true_measure)
    evalutations = my_integrand.f(ones((3,5)))


    # Accumulate AccumulateData
    class MyAccumData(AccumulateData):
        
        def __init__(self, stopping_criterion, integrand):
            self.stopping_criterion = stopping_criterion
            self.integrand = integrand
            self.measure = self.integrand.measure
            self.distribution = self.measure.distribution
            self.solution = None
            self.n_total = None
            super().__init__()
        
        def update_data(self):
            return


    # Stopping Criterion
    class MyStoppingCriterion(StoppingCriterion):

        def __init__(self, integrand):
            self.abs_tol = None
            self.rel_tol = None
            self.n_max = None
            self.data = MyAccumData(self, integrand)
            distribution = integrand.measure.distribution
            allowed_levels = 'single' # or 'multi'
            allowed_distribs = ['MyDiscreteDistribution']
            super().__init__(distribution, allowed_levels, allowed_distribs)
        
        def integrate(self):
            return self.data.solution, self.data

    my_stopping_criterion = MyStoppingCriterion(my_integrand)
    my_stopping_criterion.integrate()


if __name__ == '__main__':
    barebones()
