from qmcpy import *
from numpy import pi,sqrt
def m(x):
    if sqrt(x[0]**2+x[1]**2)<1 and x[0]<1 and x[1]<1:
        return 4./pi
    else:
        return 0.
K = Uniform(Lattice(2,seed=7))
tm = ImportanceSampling(objective_pdf=m, measure_to_sample_from=K)