from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian
from numpy import *


class Linear(Integrand):
    """
    $f(\\boldsymbol{x}) = \\sum_{i=1}^d x_i$

    >>> dd = Sobol(100,seed=7)
    >>> m = Gaussian(dd,mean=(-1)**arange(100),covariance=1./3)
    >>> l = Linear(m)
    >>> x = dd.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean()
    -0.0008447105940667683
    """

    def __init__(self, measure):
        """
        Args:
            measure (TrueMeasure): a TrueMeasure instance
        """
        self.measure = measure
        self.distribution = self.measure.distribution
        super(Linear,self).__init__()

    def g(self, x):
        """ See abstract method. """
        y = x.sum(1)  # Linear sum
        return y
    
    def plot(self, projection_dims=[0], n=2**7, point_size=5, color='c', show=True, out=None):
        """
        Make a scatter plot from samples. Requires dimension >= 2. 

        Args:
            projection_dims (list of ints): dimensions to project onto individual dimensions. 
                For example: projection_dims=[0,1] will make 2 plots. One with y vs x_0 and one with y vs x_1. 
            n (int): number of samples to draw as self.gen_samples(n)
            point_size (int): ax.scatter(...,s=point_size)
            color (str): ax.scatter(...,color=color)
            show (bool): show plot or not? 
            out (str): file name to output image. If None, the image is not output

        Returns: 
            tuple: fig,ax from `fig,ax = matplotlib.pyplot.subplots(...)`
        """
        x = self.distribution.gen_samples(n)
        y = self.f(x).squeeze()
        from matplotlib import pyplot
        pyplot.rc('font', size=16)
        pyplot.rc('legend', fontsize=16)
        pyplot.rc('figure', titlesize=16)
        pyplot.rc('axes', titlesize=16, labelsize=16)
        pyplot.rc('xtick', labelsize=16)
        pyplot.rc('ytick', labelsize=16)
        l = len(projection_dims)
        fig,ax = pyplot.subplots(nrows=1,ncols=l, figsize=(5*l,6.25))
        if l==1: ax = [ax]
        for p in range(l):
            d = projection_dims[p]
            ax[p].scatter(x[:,d],y,color=color,s=point_size)
            ax[p].set_xlabel('$x_{i,%d}$'%d)
            if p==0: ax[p].set_ylabel('$f(x)$')
            if self.distribution.mimics == 'StdUniform':
                ax[p].set_xlim([0,1])
                ax[p].set_xticks([0,1])
            elif self.distribution.mimics == 'StdGaussian':
                ax[p].set_xlim([-3,3])
                ax[p].set_xticks([-3,3])
        s = '$2^{%d}$'%log2(n) if log2(n)%1==0 else '%d'%n
        fig.suptitle(s+' Linear Evaluations')
        if l==1: pyplot.gcf().subplots_adjust(left=0.2)
        if out: pyplot.savefig(out,dpi=250)
        if show: pyplot.show()
        return fig,ax
