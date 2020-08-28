from ._true_measure import TrueMeasure
from ..util import TransformError, DimensionError
from ..discrete_distribution import Sobol
from numpy import *
from scipy.stats import norm


class Uniform(TrueMeasure):
    """
    >>> u = Uniform(Sobol(2,seed=7),lower_bound=1,upper_bound=2)
    >>> u
    Uniform (TrueMeasure Object)
        lower_bound     [1 1]
        upper_bound     [2 2]
    >>> u.gen_samples(n_min=4,n_max=8)
    array([[1.857, 1.258],
           [1.357, 1.758],
           [1.607, 1.508],
           [1.107, 1.008]])
    >>> u.set_dimension(4)
    >>> u
    Uniform (TrueMeasure Object)
        lower_bound     [1 1 1 1]
        upper_bound     [2 2 2 2]
    >>> u.gen_samples(n_min=2,n_max=4)
    array([[1.732, 1.133, 1.601, 1.354],
           [1.232, 1.633, 1.101, 1.854]])
    >>> u2 = Uniform(Sobol(2),lower_bound=[-.5,0],upper_bound=[1,3])
    >>> u2
    Uniform (TrueMeasure Object)
        lower_bound     [-0.5  0. ]
        upper_bound     [1 3]
    >>> u2.pdf([0,1])
    0.2222222222222222
    """

    parameters = ['lower_bound', 'upper_bound']
    
    def __init__(self, distribution, lower_bound=0., upper_bound=1.):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        self.distribution = distribution
        self.d = self.distribution.dimension
        if isscalar(lower_bound):
            lower_bound = tile(lower_bound,self.d)
        if isscalar(upper_bound):
            upper_bound = tile(upper_bound,self.d)
        self.lower_bound = array(lower_bound)
        self.upper_bound = array(upper_bound)
        if len(self.lower_bound)!=self.d or len(self.upper_bound)!=self.d:
            raise DimensionError('upper bound and lower bound must be of length dimension')
        super(Uniform,self).__init__()
    
    def pdf(self,x):
        """ See abstract class. """
        return 1./( prod(self.upper_bound-self.lower_bound) )

    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear Uniform
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
            ndarray: samples from the DiscreteDistribution transformed to mimic Uniform.
        """
        if self.distribution.mimics == 'StdGaussian':
            # CDF then stretch
            mimic_samples = norm.cdf(samples) * (self.upper_bound - self.lower_bound) + self.lower_bound
        elif self.distribution.mimics == "StdUniform":
            # stretch samples
            mimic_samples = samples * (self.upper_bound - self.lower_bound) + self.lower_bound
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Uniform'%self.distribution.mimics)
        return mimic_samples

    def _transform_g_to_f(self, g):
        """ See abstract method. """
        def f(samples, *args, **kwargs):
            z = self._tf_to_mimic_samples(samples)
            y = g(z, *args, **kwargs)
            return y
        return f

    def gen_samples(self, *args, **kwargs):
        """ See abstract method. """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples
    
    def set_dimension(self, dimension):
        """ See abstract method. """
        l = self.lower_bound[0]
        u = self.upper_bound[0]
        if not (all(self.lower_bound==l) and all(self.upper_bound==u)):
            raise DimensionError('In order to change dimension of uniform measure the '+\
                'lower bounds must all be the same and the upper bounds must all be the same')
        self.distribution.set_dimension(dimension)
        self.d = dimension
        self.lower_bound = tile(l,self.d)
        self.upper_bound = tile(u,self.d)
    
    def plot(self, dim_x=0, dim_y=1, n=2**7, point_size=5, color='c', show=True, out=None):
        """
        Make a scatter plot from samples. Requires dimension >= 2. 

        Args:
            dim_x (int): index of first dimension to be plotted on horizontal axis. 
            dim_y (int): index of the second dimension to be plotted on vertical axis.
            n (int): number of samples to draw as self.gen_samples(n)
            point_size (int): ax.scatter(...,s=point_size)
            color (str): ax.scatter(...,color=color)
            show (bool): show plot or not? 
            out (str): file name to output image. If None, the image is not output

        Return: 
            tuple: fig,ax from `fig,ax = matplotlib.pyplot.subplots(...)`
        """
        if self.dimension < 2:
            raise ParameterError('Plotting a Uniform instance requires dimension >=2. ')
        x = self.gen_samples(n)
        from matplotlib import pyplot
        pyplot.rc('font', size=16)
        pyplot.rc('legend', fontsize=16)
        pyplot.rc('figure', titlesize=16)
        pyplot.rc('axes', titlesize=16, labelsize=16)
        pyplot.rc('xtick', labelsize=16)
        pyplot.rc('ytick', labelsize=16)
        fig,ax = pyplot.subplots()
        x_low = self.lower_bound[dim_x]
        x_high = self.upper_bound[dim_x]
        y_low = self.lower_bound[dim_y]
        y_high = self.upper_bound[dim_y]
        ax.set_xlim([x_low,x_high])
        ax.set_xticks([x_low,x_high])
        ax.set_ylim([y_low,y_high])
        ax.set_yticks([y_low,y_high])
        ax.set_xlabel('$x_{i,%d}$'%dim_x)
        ax.set_ylabel('$x_{i,%d}$'%dim_y)
        ax.scatter(x[:,dim_x],x[:,dim_y],color=color,s=point_size)
        s = '$2^{%d}$'%log2(n) if log2(n)%1==0 else '%d'%n
        ax.set_title(s+' Uniform Samples')
        fig.tight_layout()
        if out: pyplot.savefig(out,dpi=250)
        if show: pyplot.show()
        return fig,ax
