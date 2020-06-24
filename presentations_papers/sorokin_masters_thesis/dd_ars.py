''' Code '''
from qmcpy import *
def m(x):
    if x<0 or x>=1:
        return 0.
    x = x if x<.5 else 1-x
    return 16.*x/3 if x<1./4 else 4./3
K = Gaussian(IIDStdGaussian(1,seed=7),mean=.5)
ars = AcceptanceRejectionSampling(
    objective_pdf = m,
    measure_to_sample_from = K)
x = ars.gen_samples(2**13)
''' Plots '''
from matplotlib import pyplot
pyplot.rc('font', size=11)
pyplot.rc('axes', titlesize=11)
pyplot.rc('axes', labelsize=11)
pyplot.rc('xtick', labelsize=11)
pyplot.rc('ytick', labelsize=11)
pyplot.rc('legend', fontsize=11)
pyplot.rc('figure', titlesize=11)
from numpy import arange, array
z = arange(0,1.01,.01)
fig,ax = pyplot.subplots(nrows=1, ncols=1, figsize=(6,5))
ax.hist(x,bins='auto',density=True,color='c')
ax.plot(z,array([m(z_i) for z_i in z]).squeeze(),color='r',label='$m(x)$')
ax.plot(z,ars.c*array([K.pdf(z_i) for z_i in z]).squeeze(),color='b',label='$c*k(x)$')
ax.set_xlim([0,1])
ax.set_xticks([0,1])
ax.set_xlabel('$x$')
ax.set_ylim([0,1.75])
ax.set_yticks([0,1,1.5])
ax.set_ylabel('probability density')
ax.legend(ncol=3,frameon=False)
ax.spines['top'].set_visible(False)
fig.suptitle('Histogram of Sample Density')
pyplot.tight_layout()
pyplot.savefig('presentations_papers/sorokin_masters_thesis/figs/dd_ars.png',dpi=250)
pyplot.show()