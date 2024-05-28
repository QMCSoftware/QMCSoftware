""" Plot functions used by Discrete Distribution and True Measure"""
import qmcpy as qp
from numpy import*
import os
def plot_proj(sampler, n = 64, d_horizontal = 1, d_vertical = 2,math_ind = True, marker_size = 5, figfac = 5, **kwargs):
    """
    Args:
        sampler: the Discrete Distribution or the True Measure Object to be plotted
        n (int or list): the number of samples or a list of samples(used for extensibility) to be plotted. 
            Default value is 64
        d_horizontal (int or list): the dimension or list of dimensions to be plotted on the horizontal axes. 
            Default value is 1 (1st dimension).
        d_vertical (int or list): the dimension or list of dimensions to be plotted on the vertical axes. 
            Default value is 2 (2nd dimension).
        math_ind : setting it true will enable user to pass in math indices. 
            Default value is true, so user is required to pass in math indices.
        marker_size: the marker size in points**2(typographic points are 1/72 in.).
            Default value is 5.
        figfac: the figure size factor. Default value is 5.
        **kwargs : Any extra features the user would like to see in the plots
    """
    try:
        import matplotlib.pyplot as plt
        dir_path = os.path.dirname(os.path.realpath(__file__))
        plt.style.use(os.path.join(dir_path, "../qmcpy.mplstyle"))
        from matplotlib import colors
    except:
        raise ImportError("Missing matplotlib.pyplot as plt, Matplotlib must be intalled to run plot_proj function")
    n = atleast_1d(n)
    d_horizontal = atleast_1d(d_horizontal)
    d_vertical = atleast_1d(d_vertical)
    samples = sampler.gen_samples(n[n.size - 1])    
    d = samples.shape[1]
    fig, ax = plt.subplots(nrows=d_horizontal.size, ncols=d_vertical.size, figsize=(figfac*d_horizontal.size, figfac*d_vertical.size),squeeze=False)                    
    fig.tight_layout(pad=2)
    
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
                        x_label_num = d_horizontal[i]
                        y_label_num = d_vertical[j]
                    else:
                        x = d_horizontal[i]
                        y = d_vertical[j]
                        x_label_num = d_horizontal[i] + 1
                        y_label_num = d_vertical[j] + 1
                    
                    if(isinstance(sampler,qp.DiscreteDistribution)):
                        ax[i,j].set_xlim([0,1])
                        ax[i,j].set_ylim([0,1])
                        ax[i,j].set_xticks([0,1/4,1/2,3/4,1])
                        ax[i,j].set_yticks([0,1/4,1/2,3/4,1])
                        ax[i,j].set_aspect(1)
                        x_label = r'$x_{i%d}$'%(x_label_num)
                        y_label = r'$x_{i%d}$'%(y_label_num)
                    else:
                        x_label = r'$t_{i%d}$'%(x_label_num)
                        y_label = r'$t_{i%d}$'%(y_label_num)    
                    
                    ax[i,j].set_xlabel(x_label,fontsize = 15)
                    if(d > 1):
                        ax[i,j].set_ylabel(y_label,fontsize = 15)
                        y_axis = samples[n_min:n_max,y]
                    else:
                        y_axis = []
                        for h in range(n_max-n_min):
                            y_axis.append(0.5)
                    ax[i,j].scatter(samples[n_min:n_max,x],y_axis,s=marker_size,color=colors[m],label='n_min = %d, n_max = %d'%(n_min,n_max),**kwargs)
                    n_min = n[m]
    fig.suptitle('Projection of Samples',fontsize = 20)
    return fig, ax