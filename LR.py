from qmcpy.integrand._integrand import Integrand
from qmcpy import Sobol, Uniform, CubQMCSobolG, CubMCCLT
import numpy
from qmcpy import *

class LogisticRegression(Integrand):
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
        
        super(LogisticRegression, self).__init__()
        
    def g(self, x):
        product = 1
        m, d = self.s.shape
        a, b = x.shape
        for c in range(0, a):
            for i in range(0, m):
                total = x[c][0]
                for j in range(1, d+1):
                    total  = (x[c][j]* self.s[i][j-1]) + total
                value1 = numpy.exp(total)
                value2 = value1 / (1 + value1)
                product = product * (value2)**(self.t[i])
                product = product * (1 - value2)**(1-self.t[i])
                return product

s = numpy.array([[0]])
t = numpy.array([1])

no, dim = s.shape

k = LogisticRegression(IIDStdUniform(dim+1,seed=7), s_matrix = s, t = t)
solution, data = CubMCCLT(k, abs_tol = .00001).integrate()
print(solution)

k1 = LogisticRegression(Sobol(dim,seed=7), s_matrix = s, t = t)
#print(k.f(numpy.array([[0, 1/2, 1]]).T))
solution1, data1 = CubQMCSobolG(k1, abs_tol = .00001).integrate()
print(solution1)

my_instance = LogisticRegression(sampler = Sobol(dimension=dim+1, seed = 7), s_matrix = s, t = t)
p = my_instance.discrete_distrib.gen_samples(n_min=0, n_max=1024)
y = my_instance.f(p)
print(y.mean())
