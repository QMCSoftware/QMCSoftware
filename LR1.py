from qmcpy.integrand.LR import LR
from sys import meta_path
import numpy
from numpy.linalg import norm as norm
from qmcpy import *

s = numpy.array([
    [109/30, 2.58], 
    [137/30, 4.88],
    [31/30, 3.34],
    [59/30, 2.75],
    [31/15, 1.9],
    [1.55, 2.11],
    [23/6, 4.18],
    [37/12, 3.3],
    [1.3, 1.3],
    [73/30, 2.73],
    [181/60, 2.38],
    [127/30, 6.37],
    [79/30, 2.53],
    [6.15, 3.41],
    [101/60, 1.59],
    [31/15, 2.96],
    [25/12, 2.63],
    [5.5, 3.1],
    [77/15, 4.75],
    [35/12, 3.38],
    [4.1, 5.58]])
t = numpy.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1])

no, dim = s.shape

k = LR(IIDStdUniform(dim+1,seed=8), s_matrix = s, t = t)
solution, data = CubMCCLT(k, abs_tol = .001).integrate()
print(data)
print(" ")
k1 = LR(Sobol(dim+1,seed=8), s_matrix = s, t = t)
solution1, data1 = CubQMCSobolG(k1, abs_tol = .001).integrate()
print(data1)
print(" ")
my_instance = LR(sampler = Sobol(dimension=dim+1, seed = 7), s_matrix = s, t = t)
p = my_instance.discrete_distrib.gen_samples(n_min=0, n_max=1024)
y = my_instance.f(p)
print(y.mean())