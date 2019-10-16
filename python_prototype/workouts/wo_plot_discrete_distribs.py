import numpy as np
import matplotlib.pyplot as plt

from qmcpy.discrete_distribution import IIDStdUniform,IIDStdGaussian,Lattice,Sobol

n = 100

def iid_scatters():
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for i,(distrib,color,lims) in enumerate(
            zip([IIDStdUniform(rng_seed=7),IIDStdGaussian(rng_seed=7)],['b','r'],[[0,1],[-3,3]])): 
        samples = distrib.gen_samples(1,n,2).squeeze()
        ax[i].scatter(samples[:,0],samples[:,1],color=color)
        ax[i].set_xlabel('$x_0$')
        ax[i].set_ylabel('$x_1$')
        ax[i].set_xlim(lims), ax[i].set_ylim(lims)
    ax[0].set_title('IID Standard Uniform')
    ax[1].set_title('IID Standard Gaussian')
    plt.tight_layout()
    plt.show()
    fig.savefig('outputs/scatters_iid.png',dpi=200)

def lds_scatters():
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    for i,(distrib,color) in enumerate(
            zip([Lattice(rng_seed=7),Sobol(rng_seed=7)],['g','c'])): 
        samples = distrib.gen_samples(1,n,2).squeeze()
        ax[i].scatter(samples[:,0],samples[:,1],color=color)
        ax[i].set_xlabel('$x_0$')
        ax[i].set_ylabel('$x_1$')
        ax[i].set_xlim([0,1]), ax[i].set_ylim([0,1])
    ax[0].set_title('Shifted Lattice')
    ax[1].set_title('Scrambled Sobol')
    plt.tight_layout()
    plt.show()
    fig.savefig('outputs/scatters_lds.png',dpi=200)

if __name__ == '__main__':
    iid_scatters()
    lds_scatters()