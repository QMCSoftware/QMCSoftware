from copy import deepcopy
from numpy import arange
import matplotlib.pyplot as plt

from qmcpy.discrete_distribution import IIDStdUniform,IIDStdGaussian,Lattice,Sobol
from qmcpy.true_measure import Uniform,Gaussian,BrownianMotion

n = 128

def iid_scatters():
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
    for i,(distrib,color,lims,title) in enumerate(
            zip([IIDStdUniform(rng_seed=7),IIDStdGaussian(rng_seed=7)],
            ['b','r'],[[0,1],[-2.5,2.5]],['IID Standard Uniform','IID Standard Gaussian'])): 
        samples = distrib.gen_samples(1,n,2).squeeze()
        ax[i].scatter(samples[:,0],samples[:,1],color=color)
        ax[i].set_xlabel('$x_1$')
        ax[i].set_ylabel('$x_2$')
        ax[i].set_xlim(lims)
        ax[i].set_ylim(lims)
        ax[i].set_aspect('equal')
        ax[i].set_title(title)
    plt.tight_layout()
    plt.show()
    fig.savefig('outputs/scatters_iid.png',dpi=200)

def lds_scatters():
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
    for i,(distrib,color,title) in enumerate(
            zip([Lattice(rng_seed=7),Sobol(rng_seed=7)],['g','c'],['Shifted Lattice','Scrambled Sobol'])): 
        samples = distrib.gen_samples(1,n,2).squeeze()
        ax[i].scatter(samples[:,0],samples[:,1],color=color)
        ax[i].set_xlabel('$x_1$')
        ax[i].set_ylabel('$x_2$')
        ax[i].set_xlim([0,1])
        ax[i].set_ylim([0,1])
        ax[i].set_aspect('equal')
        ax[i].set_title(title)
    plt.tight_layout()
    plt.show()
    fig.savefig('outputs/scatters_lds.png',dpi=200)

def plot_grid_transforms():
    true_measures = [Uniform(2),Gaussian(2),BrownianMotion(dimension=2,time_vector=[arange(1/2,3/2,1/2)])]
    discrete_distribs = [IIDStdUniform(rng_seed=7),IIDStdGaussian(rng_seed=7),Lattice(rng_seed=7),Sobol(rng_seed=7)]
    colors = ['r','g','b']
    lims = [[0,1],[-2.5,2.5],[-2.5,2.5]]
    for true_measure,lim,color in zip(true_measures,lims,colors):
        tm_name = type(true_measure).__name__
        fig,ax = plt.subplots(nrows=1, ncols=4, figsize=(16,5))
        for j,discrete_distrib in enumerate(discrete_distribs):
            tm_obj = deepcopy(true_measure)
            dd_obj = deepcopy(discrete_distrib)
            tm_obj.transform_generator(dd_obj)
            tm_samples = tm_obj[0].gen_true_measure_samples(1,n).squeeze()
            ax[j].scatter(tm_samples[:,0],tm_samples[:,1],color=color)
            ax[j].set_xlabel('$x_1$')
            ax[j].set_ylabel('$x_2$')
            ax[j].set_xlim(lim)
            ax[j].set_ylim(lim)
            ax[j].set_aspect('equal')
            ax[j].set_title(type(discrete_distrib).__name__)
        fig.suptitle('Transformed to %s'%tm_name)
        plt.tight_layout()
        fig.savefig('outputs/scatters_transform_to_%s.png'%tm_name,dpi=200)
        plt.show()


if __name__ == '__main__':
    iid_scatters()
    lds_scatters()
    plot_grid_transforms()