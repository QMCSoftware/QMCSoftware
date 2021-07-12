from qmcpy.true_measure.uniform import Uniform
from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError
import numpy
from numpy.linalg import norm as norm

class LR(Integrand):
    def __init__(self, sampler, s_matrix, t):
        self.true_measure = Uniform(sampler)
        m, d = s_matrix.shape
        check1 = True
        if m != len(t):
            check1 = False
        if check1 == True:
            self.s = s_matrix
        else:
            print("s_matrix must have the same amount of rows as the amount of elements in t")
        check = True
        for i in range(len(t)):
            if 0 > t[i] or t[i] > 1:
                check = False
        if check == True:
            self.t = t
        else:
            print("for all 't_i', 0 <= t_i <= 1")
        
        super(LR, self).__init__()
        
    def g(self, x):
        m, d = self.s.shape
        a, b = x.shape
        values = []
        for c in range(0, a):
            product = 1
            for i in range(0, m):
                total = x[c][0]
                for j in range(1, d+1):
                    total  = (x[c][j] * self.s[i][j-1]) + total
                value1 = numpy.exp(total)
                value2 = value1 / (1 + value1)
                product = product * (value2)**(self.t[i])
                product = product * (1 - value2)**(1-self.t[i])
            values = values + [product]
        values = numpy.array(values)
        return values