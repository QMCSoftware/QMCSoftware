''' Code '''
from qmcpy import *
u = Uniform(Sobol(2,seed=7), lower_bound=[-2,0], upper_bound=[2,4])
x_u = u.gen_mimic_samples(2**7)
g = Gaussian(Sobol(2,seed=7), mean=[3.,2.], covariance=[[9.,5.], [5.,4.]])
x_g = g.gen_mimic_samples(2**7)
bm = BrownianMotion(Sobol(2,seed=7), drift=2)
x_bm = bm.gen_mimic_samples(2**5)
''' Plots '''
from matplotlib import pyplot
pyplot.rc('font', size=11)
pyplot.rc('axes', titlesize=11)
pyplot.rc('axes', labelsize=11)
pyplot.rc('xtick', labelsize=11)
pyplot.rc('ytick', labelsize=11)
pyplot.rc('legend', fontsize=11)
pyplot.rc('figure', titlesize=11)
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from numpy import hstack, zeros
fig,ax = pyplot.subplots(nrows=1, ncols=3, figsize=(15,5.5))
# uniform
ax[0].scatter(x_u[:,0],x_u[:,1],color='r')
ax[0].set_xlim([-2.,2.])
ax[0].set_xticks([-2,2])
ax[0].set_xlabel('$x_{i1}$')
ax[0].set_ylim([0.,4.])
ax[0].set_yticks([0,4])
ax[0].set_ylabel('$x_{i2}$')
ax[0].set_aspect(1)
ax[0].set_title('$\mathcal{U}([-2,0],[2,4])$')
# Gaussian
ax[1].scatter(x_g[:,0],x_g[:,1],color='g')
ax[1].set_xlim([-6.,12.])
ax[1].set_xticks([-6,12])
ax[1].set_xlabel('$x_{i1}$')
ax[1].set_ylim([-2.,6.])
ax[1].set_yticks([-2,6])
ax[1].set_ylabel('$x_{i2}$')
ax[1].set_aspect(18./8)
ax[1].set_title(r'$\mathcal{N}([3,2],\begin{bmatrix}9&5\\5&4\end{bmatrix})$')
# Brownian motion
tv = hstack((0,bm.time_vector))
for i in range(x_bm.shape[0]):
    ax[2].plot(tv,hstack((0,x_bm[i,:])))
ax[2].set_xlim([0.,1.])
ax[2].set_xticks(tv)
ax[2].set_xlabel('$t$')
ax[2].set_ylim([-1,5])
ax[2].set_yticks([-1,5])
ax[2].set_ylabel('$x_{i,2t}$')
ax[2].set_title('Discrete Brownain Motion\nwith Monitoring Times $t=[1/2,1]$')
ax[2].set_aspect(1./6)
# meta info
pyplot.tight_layout()
pyplot.savefig('presentations_papers/sorokin_masters_thesis/figs/tm1.png',dpi=250)
pyplot.show()