''' Code '''
from qmcpy import *
x_l = Lattice(dimension=2, backend="GAIL", randomize=True, seed=7).gen_samples(2**7)
x_s = Sobol(dimension=2, backend="QRNG", randomize=True, seed=7).gen_samples(2**7)
x_h = Halton(dimension=2, backend="Owen", generalize=True, seed=7).gen_samples(2**7) 
x_k = Korobov(dimension=2, generator=[7,13], randomize=True, seed=7).gen_samples(2**7)
''' Plots '''
from matplotlib import pyplot
fs = 25
pyplot.rc('font', size=fs)
pyplot.rc('axes', titlesize=fs)
pyplot.rc('axes', labelsize=fs)
pyplot.rc('xtick', labelsize=fs)
pyplot.rc('ytick', labelsize=fs)
pyplot.rc('legend', fontsize=fs)
pyplot.rc('figure', titlesize=fs)
fig,ax = pyplot.subplots(nrows=1, ncols=4, figsize=(20,5.1))
ax[0].scatter(x_l[:,0],x_l[:,1],color='r')#[(0,0,1)])
ax[0].set_title('Shifted Lattice')
ax[1].scatter(x_s[:,0],x_s[:,1],color='g')#[(0,0,1)])
ax[1].set_title("Shifted Sobol'")
ax[2].scatter(x_h[:,0],x_h[:,1],color='b')#[(0,0,1)])
ax[2].set_title('Generalized Halton')
ax[3].scatter(x_k[:,0],x_k[:,1],color='m')#[(0,0,1)])
ax[3].set_title('Randomized Korobov')
# meta info
for i in range(4):
    ax[i].set_xlim([0,1])
    ax[i].set_xticks([0,1])
    ax[i].set_ylim([0,1])
    ax[i].set_yticks([0,1])
    ax[i].set_aspect(1)
    ax[i].set_xlabel('$x_{i1}$')
    ax[i].set_ylabel('$x_{i2}$')
pyplot.tight_layout()
pyplot.savefig('presentations_papers/sorokin_masters_thesis/figs/dd_ld.png',
    dpi=500,transparent=True)
pyplot.show()
