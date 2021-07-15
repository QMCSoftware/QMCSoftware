from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian, Lebesgue, Uniform
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError
from numpy import *

class LR(Integrand):

    """
    $f(\\boldsymbol{t}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{t} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1./2.

    >>> s = numpy.array([[0]])
    >>> t = numpy.array([1])
    >>> no, dim = s.shape
    >>> my_instance = LR(sampler = Sobol(dimension=dim+1, seed = 7), s_matrix = s, t = t)
    >>> p = my_instance.discrete_distrib.gen_samples(n_min=0, n_max=2**10)
    >>> y = my_instance.f(p)
    >>> print(y.mean())
    0.6201145005168884
    >>> my_instance.true_measure
    Uniform (TrueMeasure Object)
        lower_bound     0
        upper_bound     1
    (qmcpy) 
    """
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
            if 0 != t[i] and t[i] != 1:
                check = False
        if check == True:
            self.t = t
        else:
            print("for all 't_i', t_i can only equal 0 or 1")
        self.dprime = 1
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
                value1 = exp(total)
                value2 = value1 / (1 + value1)
                product = product * (value2)**(self.t[i])
                product = product * (1 - value2)**(1-self.t[i])
            values = values + [product]
        values = array(values)
        return values
