''' Code '''
from qmcpy import *
from numpy import log
l = 1.5 # lambda
exp_dd = InverseCDFSampling(Sobol(2,seed=7), lambda u: -log(1-u)/l)
x = exp_dd.gen_samples(2**7)
''' Plots '''
from matplotlib import pyplot
pyplot.rc('font', size=11)
pyplot.rc('axes', titlesize=11)
pyplot.rc('axes', labelsize=11)
pyplot.rc('xtick', labelsize=11)
pyplot.rc('ytick', labelsize=11)
pyplot.rc('legend', fontsize=11)
pyplot.rc('figure', titlesize=11)
fig,ax = pyplot.subplots(nrows=1, ncols=1, figsize=(5,5))
ax.scatter(x[:,0],x[:,1],color='r')
ax.set_xlim([0,4])
ax.set_xticks([0,4])
ax.set_xlabel('$x_{i1}$')
ax.set_ylim([0,4])
ax.set_yticks([0,4])
ax.set_ylabel('$x_{i2}$')
ax.set_aspect(1)
ax.set_title("Exp($\lambda=$3/2) Samples by Inverse CDF Transform\nto Shifted Sobol' Samples")
pyplot.tight_layout()
pyplot.savefig('presentations_papers/sorokin_masters_thesis/figs/dd_exp.png',dpi=250)
pyplot.show()