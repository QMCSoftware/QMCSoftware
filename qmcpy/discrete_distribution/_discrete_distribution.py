from ..util import ParameterError, MethodImplementationError, _univ_repr, DimensionError
from numpy import *


class DiscreteDistribution(object):
    """ Discrete Distribution abstract class. DO NOT INSTANTIATE. """

    def __init__(self, dimension, seed):
        """ 
        Args:
            dimension (int or ndarray): dimension of the generator. 
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence. 
                    Note that this is not relevent for IID generators.
            seed (int or numpy.random.SeedSequence): seed to create random number generator
        """
        prefix = 'A concrete implementation of DiscreteDistribution must have '
        if not hasattr(self, 'mimics'):
            raise ParameterError(prefix + 'self.mimcs (measure mimiced by the distribution)')
        if not hasattr(self, 'low_discrepancy'):
            raise ParameterError(prefix + 'self.low_discrepancy')
        if not hasattr(self,'parameters'):
            self.parameters = []
        if not hasattr(self,'d_max'):
            raise ParameterError(prefix+ 'self.dmax')
        if isinstance(dimension,list) or isinstance(dimension,ndarray):
            self.dvec = array(dimension)
            self.d = len(self.dvec)
        else:
            self.d = dimension
            self.dvec = arange(self.d)
        if any(self.dvec>self.d_max):
            raise ParameterError('dimension greater than max dimension %d'%self.d_max)
        self._base_seed = seed if isinstance(seed,random.SeedSequence) else random.SeedSequence(seed)
        self.entropy = self._base_seed.entropy
        self.spawn_key = self._base_seed.spawn_key
        self.rng = random.Generator(random.SFC64(self._base_seed))

    def gen_samples(self, *args):
        """
        ABSTRACT METHOD to generate samples from this discrete distribution.

        Args:
            args (tuple): tuple of positional argument. See implementations for details

        Returns:
            ndarray: n x d array of samples
        """
        raise MethodImplementationError(self, 'gen_samples')

    def pdf(self, x):
        """ ABSTRACT METHOD to evaluate pdf of distribution the samples mimic at locations of x. """
        raise MethodImplementationError(self, 'pdf')

    def spawn(self, s=1, dimensions=None):
        """
        Spawn new instances of the current discrete distribution but with new seeds and dimensions. 
        Developed for multi-level and multi-replication (Q)MC algorithms.
        
        Args:
            s (int): number of spawn
            dimensions (ndarray): length s array of dimension for each spawn. Defaults to current dimension
        
        Return: 
            list: list of DiscreteDistribution instances with new seeds and dimensions
        """
        if dimensions is None:
            dimensions = tile(self.d,s)
        elif isscalar(dimensions) and dimensions%1==0:
            dimensions = tile(dimensions,s)
        elif len(dimensions)==s:
            dimensions = array(dimensions)
        else:
            raise ParameterError("invalid spawn dimensions, must be None, int, or length s ndarray")
        child_seeds = self._base_seed.spawn(s)
        return self._spawn(s,child_seeds,dimensions)
    
    def _spawn(self, s, child_seeds, dimensions):
        """
        ABSTRACT METHOD, used by self.spawn
        
        Args:
            s (int): number of spawns
            child_seeds (numpy.random.SeedSequence): length s array of seeds for each spawn
            dimensions (ndarray): lenth s array of dimensions for each spawn
        
        Return: 
            list: list of DiscreteDistribution instances with new seeds and dimensions
        """
        raise MethodImplementationError(self, '_spawn')
        
    def __repr__(self):
        return _univ_repr(self, "DiscreteDistribution", ['d']+self.parameters+['entropy','spawn_key'])
    
    def __call__(self, *args, **kwargs):
        if len(args)>2 or len(args)==0:
            raise Exception('''
                expecting 1 or 2 arguments:
                    1 argument corresponds to n, the number of smaples to generate. In this case n_min=0 and n_max=n for LD sequences
                    2 arguments corresponds to n_min and n_max. Note this is incompatible with IID generators which only expect 1 argument.
                ''')
        if len(args) == 1:
            return self.gen_samples(n=args[0])
        else:
            return self.gen_samples(n_min=args[0],n_max=args[1])

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
            fig,ax (tuple) from fig,ax = matplotlib.pyplot.subplots(...)
        """
        if self.dimension < 2:
            raise ParameterError('Plotting a discrete distribution requires dimension >=2. ')
        x = self.gen_samples(n)
        from matplotlib import pyplot
        pyplot.rc('font', size=16)
        pyplot.rc('legend', fontsize=16)
        pyplot.rc('figure', titlesize=16)
        pyplot.rc('axes', titlesize=16, labelsize=16)
        pyplot.rc('xtick', labelsize=16)
        pyplot.rc('ytick', labelsize=16)
        if self.mimics == 'StdUniform':
            fig,ax = pyplot.subplots(figsize=(5,5))
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            ax.set_aspect(1)
        elif self.mimics == 'StdGaussian':
            fig,ax = pyplot.subplots(figsize=(5,5))
            ax.set_xlim([-3,3])
            ax.set_ylim([-3,3])
            ax.set_xticks([-3,3])
            ax.set_yticks([-3,3])
            ax.set_aspect(1)
        else:
            fig,ax = pyplot.subplots()
        ax.set_xlabel('$x_{i,%d}$'%dim_x)
        ax.set_ylabel('$x_{i,%d}$'%dim_y)
        ax.scatter(x[:,dim_x],x[:,dim_y],color=color,s=point_size)
        s = '$2^{%d}$'%log2(n) if log2(n)%1==0 else '%d'%n
        ax.set_title(s+' %s Samples'%type(self).__name__)
        fig.tight_layout()
        if out: pyplot.savefig(out,dpi=250)
        if show: pyplot.show()
        return fig,ax
