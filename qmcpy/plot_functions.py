""" Plot functions used by Discrete Distribution and True Measure"""
import qmcpy as qp
from numpy import*
from .discrete_distribution import *
from .true_measure import *
def plot_proj(n,sampler, d_horizontal = 0, d_vertical = 1,axis=None, math_ind = False, **kwargs):
    """
    Args:
        n(int or array): n is the number of samples that will be plotted or a list of samples(used for extensible point sets)
        sampler: The Discrete Distribution object or the True Measure Object for which this function will plot the number of samples or a list of samples as determined by n
        d_horizontal (int or array): d_horizontal is a list of dimensions to be plotted on the horizontal axes. Possible input values are from 0 to d-1. Default value is 0 (1st dimension).
        d_vertical (int or array): d_vertical is a list of dimensions to be plotted on the vertical axes for each corresponding element in d_horizontal. Default value is 1 (2nd dimension).
        math_ind : If user wants to pass in the math indices, set it true. By default it is false, so by default this method takes in pyton indices.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except:
        raise ImportError("Missing matplotlib.pyplot as plt, Matplotlib must be intalled to run plot_proj function")
    n = atleast_1d(n)
    d_horizontal = atleast_1d(d_horizontal)
    d_vertical = atleast_1d(d_vertical)
    samples = sampler.gen_samples(n[n.size - 1])    
    d = samples.shape[1]
    assert d>=2 

    if axis is None:
        fig, ax = plt.subplots(nrows=d_horizontal.size, ncols=d_vertical.size, figsize=(3*d, 3*d),squeeze=False)                    
        fig.tight_layout(pad=2)
    else:
        ax = axis 
        fig = plt.figure()
        assert (ax.shape[0] >= d_horizontal.size) and (ax.shape[1] >= d_vertical.size)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(d_horizontal.size):            
        for j in range(d_vertical.size):
                n_min = 0
                for m in range(n.size):
                    n_max = n[m]  
                    if(d_horizontal[i] == d_vertical[j]):
                        ax[i,j].remove()  
                        break
                    if(math_ind is True):
                        x = d_horizontal[i] - 1
                        y = d_vertical[j] - 1
                        x_label = d_horizontal[i]
                        y_label = d_vertical[j]
                    else:
                        x = d_horizontal[i]
                        y = d_vertical[j]
                        x_label = d_horizontal[i] + 1
                        y_label = d_vertical[j] + 1
                    ax[i,j].set_xlabel(r'$x_{i%d}$'%(x_label)); ax[i,j].set_ylabel(r'$x_{i%d}$'%(y_label))
                    ax[i,j].scatter(samples[n_min:n_max,x],samples[n_min:n_max,y],s=5,color=colors[m],label='n_min = %d, n_max = %d'%(n_min,n_max),**kwargs)
                    n_min = n[m]
    return fig, ax