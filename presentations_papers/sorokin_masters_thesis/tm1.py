
from qmcpy import *
mu = [3.,2.]
sigma = [[9. , 5.],
         [5. , 4.]]
dd = Sobol(2, seed=7) # discrete distribution
tm = Gaussian(dd, mean=mu, covariance=sigma) # true measure
x = tm.gen_mimic_samples(2**7)

from matplotlib import pyplot
pyplot.scatter(x[:,0],x[:,1],color='c')
pyplot.xlim([-6.,12.])
pyplot.xticks([-6.,0.,12.])
pyplot.ylim([-2.,6.])
pyplot.yticks([-2.,2.,6.])
pyplot.tight_layout()
pyplot.savefig('presentations_papers/sorokin_masters_thesis/figs/tm1.png',dpi=250)