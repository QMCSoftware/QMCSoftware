from qmcpy import *
from matplotlib import pyplot

pyplot.rc('font', size=16)          # controls default text sizes
pyplot.rc('axes', titlesize=16)     # fontsize of the axes title
pyplot.rc('axes', labelsize=16)    # fontsize of the x and y labels
pyplot.rc('xtick', labelsize=16)    # fontsize of the tick labels
pyplot.rc('ytick', labelsize=16)    # fontsize of the tick labels
pyplot.rc('legend', fontsize=16)    # legend fontsize
pyplot.rc('figure', titlesize=16)  # fontsize of the figure title


n = 64

pts_sets = [
    IIDStdUniform(2,seed=7).gen_samples(n),
    Lattice(2,seed=7).gen_samples(n)]
titles = ['$U[0,1]^2$','Shifted Lattice']
symbols = ['T','X']
output_files = ['iid_uniform_pts','lattice_pts']

for pts,title,symbol,out_f in zip(pts_sets,titles,symbols,output_files):
    fig,ax = pyplot.subplots(nrows=1, ncols=1, figsize=(5,5))
    ax.scatter(pts[:,0],pts[:,1],color='b')
    ax.set_xlabel('$%s_{i1}$'%symbol)
    ax.set_xlim([0,1])
    ax.set_xticks([0,1])
    ax.set_ylabel('$%s_{i2}$'%symbol)
    ax.set_ylim([0,1])
    ax.set_yticks([0,1])
    ax.set_title(title)
    ax.set_aspect('equal')
    pyplot.tight_layout()
    fig.savefig('blogs/why_q_in_mc/%s.png'%out_f,dpi=200)
