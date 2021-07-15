from qmcpy.integrand.LR import LR
from sys import meta_path
import numpy
from numpy.linalg import norm as norm
from qmcpy import *

s = numpy.array([[0]])
t = numpy.array([1])

no, dim = s.shape

#k = LR(IIDStdUniform(dim+1,seed=8), s_matrix = s, t = t)
#solution, data = CubMCCLT(k, abs_tol = .001).integrate()
#print(data)

#k1 = LR(Sobol(dim+1,seed=8), s_matrix = s, t = t)
#solution1, data1 = CubQMCSobolG(k1, abs_tol = .001).integrate()
#print(data1)

my_instance = LR(sampler = Sobol(dimension=dim+1, seed = 7), s_matrix = s, t = t)
p = my_instance.discrete_distrib.gen_samples(n_min=0, n_max=1024)
y = my_instance.f(p)
print(y.mean())
print(my_instance.true_measure)