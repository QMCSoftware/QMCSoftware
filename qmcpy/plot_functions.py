""" Plot functions used by Discrete Distribution and True Measure"""
import qmcpy as qp
from numpy import *
from .discrete_distribution import *
from .true_measure import *
def plot_proj(n,discrete_distribution = None,true_measure = None, d_horizontal = 0, d_vertical = 1,axis=None, math_ind = False, **kwargs):
    """
    Args:
        n(int or array): n is the number of samples that will be plotted or a list of samples(used for extensible point sets)
        discrete_distribution: The Discrete Distribution object for which this function will plot the number of samples or a list of samples as determined by n. Default value is None
        true_measure: The True Measure object for which this function will plot the number of samples or a list of samples as determined by n. Default Value is None
        d_horizontal (int or array): d_horizontal is a list of dimensions to be plotted on the horizontal axes. Possible input values are from 0 to d-1. Default value is 0 (1st dimension).
        d_vertical (int or array): d_vertical is a list of dimensions to be plotted on the vertical axes for each corresponding element in d_horizontal. Default value is 1 (2nd dimension).
        math_ind : If user wants to pass in the math indices, set it true. By default it is false, so by default this method takes in pyton indices.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except:
        raise ImportError("Missing matplotlib.pyplot as plt, Matplotlib must be intalled to run DiscreteDistribution.plot")
    n = atleast_1d(n)
    d_horizontal = atleast_1d(d_horizontal)
    d_vertical = atleast_1d(d_vertical)
    if(discrete_distribution is None and true_measure is None):
        raise ValueError("Either a Discrete Distribution Object or a True Meaure Object needs to be passed")
    elif(discrete_distribution is not None and true_measure is not None):
        raise ValueError("Either a Discrete Distribution Object or a True Meaure Object needs to be passed. Can't pass in both")
    elif(true_measure is None):
        samples = discrete_distribution.gen_samples(n[n.size - 1])
    else:
        samples = true_measure.gen_samples(n[n.size - 1])
        
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
    """l_bound = amin(samples) - 0.03*(amax(samples) - amin(samples))"""
    """h_bound = amax(samples) + 0.03*(amax(samples) - amin(samples))"""
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
                    ax[i,j].scatter(samples[n_min:n_max,x],samples[n_min:n_max,y],s=5,color=colors[m],label='n_min = %d, n_max = %d'%(n_min,n_max),**kwargs)          
                    ax[i,j].set_aspect(1)
                    ax[i,j].set_xlabel(r'$x_{i%d}$'%(x_label)); ax[i,j].set_ylabel(r'$x_{i%d}$'%(y_label))
                    x_lbound = amin(samples[x]) - 0.03*(amax(samples[x]) - amin(samples[x]))
                    x_hbound = amax(samples[x]) + 0.03*(amax(samples[x]) - amin(samples[x]))
                    y_lbound = amin(samples[y]) - 0.03*(amax(samples[y]) - amin(samples[y]))
                    y_hbound = amax(samples[y]) + 0.03*(amax(samples[y]) - amin(samples[y]))
                    ax[i,j].set_xlim([x_lbound,x_hbound]); ax[i,j].set_ylim([y_lbound,y_hbound])
                    """ax[i,j].set_xticks([-3,3]); ax[i,j].set_yticks([-3,3]) """
                    n_min = n[m]
    return fig, ax