from qmcpy import *
from matplotlib import pyplot
from numpy import *

pyplot.rc('font', size=11)
pyplot.rc('axes', titlesize=11)
pyplot.rc('axes', labelsize=11)
pyplot.rc('xtick', labelsize=11)
pyplot.rc('ytick', labelsize=11)
pyplot.rc('legend', fontsize=11)
pyplot.rc('figure', titlesize=11)

# params
n_mesh = 1002
n_ld = 2**7
# plot setup
fig,ax = pyplot.subplots(nrows=1, ncols=2, figsize=(8,4.5))
# right plot
dd = IIDStdGaussian(1,seed=7)
m = Gaussian(dd,covariance=1./2)
k = Keister(m)
curve = zeros((n_mesh-2,2),dtype=float)
curve[:,0] = linspace(-2.5,2.5,n_mesh)[1:-1]
curve[:,1] = k.f(curve[:,0].reshape((-1,1))).flatten()
ax[0].plot(curve[:,0],curve[:,1],color='k')
pts = zeros((n_ld,2),dtype=float)
pts[:,0] = dd.gen_samples(n_ld).flatten()
pts[:,1] = k.f(pts[:,0].reshape(-1,1)).flatten()
ax[0].scatter(pts[:,0],pts[:,1],s=10,color='b')
ax[0].set_xlim([-2.5,2.5])
ax[0].set_xticks([-2,0,2])
ax[0].set_ylim([0,2])
ax[0].set_yticks([0,2])
ax[0].set_aspect(5./2)
# right plot
dd = IIDStdUniform(1,seed=7)
m = Gaussian(dd,covariance=1./2)
k = Keister(m)
curve = zeros((n_mesh-2,2),dtype=float)
curve[:,0] = linspace(0,1,n_mesh)[1:-1]
curve[:,1] = k.f(curve[:,0].reshape((-1,1))).flatten()
ax[1].plot(curve[:,0],curve[:,1],color='k')
pts = zeros((n_ld,2),dtype=float)
pts[:,0] = dd.gen_samples(n_ld).flatten()
pts[:,1] = k.f(pts[:,0].reshape(-1,1)).flatten()
ax[1].scatter(pts[:,0],pts[:,1],s=10,color='b')
ax[1].set_xlim([0,1])
ax[1].set_xticks([0,1])
ax[1].set_ylim([0,2])
ax[1].set_yticks([0,2])
ax[1].set_aspect(1./2)
# plot metas
ax[0].set_xlabel('$x_1 \sim \mathcal{N}(0,1)$')
ax[0].set_ylabel('$f_1(x_1)$')
ax[1].set_xlabel('$x_2 \sim \mathcal{U}(0,1)$')
ax[1].set_ylabel('$f_2(x_2)$')
fig.suptitle('Keister Function With Various Integrand Transforms')
pyplot.tight_layout()
pyplot.savefig("presentations_papers/sorokin_masters_thesis/figs/i_keister_figs.png",dpi=500)#1000,transparent=True)
pyplot.show()