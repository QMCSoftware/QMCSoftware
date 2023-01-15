from ..util import ParameterError, MethodImplementationError, _univ_repr, DimensionError
from numpy import *
import matplotlib.pyplot as plt


class DiscreteDistribution(object):
    """ Discrete Distribution abstract class. DO NOT INSTANTIATE. """

    def __init__(self, dimension, seed):
        """
        Args:
            dimension (int or ndarray): dimension of the generator.
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence.
                Note that this is not relevant for IID generators.
            seed (int or numpy.random.SeedSequence): seed to create random number generator
        """
        prefix = 'A concrete implementation of DiscreteDistribution must have '
        if not hasattr(self, 'mimics'):
            raise ParameterError(prefix + 'self.mimcs (measure mimiced by the distribution)')
        if not hasattr(self,'low_discrepancy'):
            raise ParameterError(prefix + 'self.low_discrepancy')
        if not hasattr(self,'parameters'):
            self.parameters = []
        if not hasattr(self,'d_max'):
            raise ParameterError(prefix+ 'self.d_max')
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

    def plot(self, n, d_vertical=0, d_horizontial =1):
        """
        ABSTRACT METHOD for ploting a distrubution, you will also need to import matplotlib.pyplot as plt

        Args:
            n (int): n is the number of samples that will be plotted 

            d_vertical (int): d_vertical is the index of points that will be plotted on the vertical axis. 

            d_horizontial (int): d_horizontial is the index of points that will be plotted as the horizontial axis.

        """
        try:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
            samples = self.gen_samples(n)
            ax.scatter(samples[:, d_vertical], samples[:, d_horizontial])
            plt.show()
        except:
            raise ImportError("Missing matplotlib.pyplot as plt")
        
        return fig, ax

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
        if (isinstance(dimensions,list) or isinstance(dimensions,ndarray)) and len(dimensions)==s:
            dimensions = array(dimensions)
        elif isscalar(dimensions) and dimensions%1==0:
            dimensions = tile(dimensions,s)
        elif dimensions is None:
            dimensions = tile(self.d,s)
        else:
            raise ParameterError("invalid spawn dimensions, must be None, int, or length s ndarray")
        child_seeds = self._base_seed.spawn(s)
        return [self._spawn(child_seeds[i],dimensions[i]) for i in range(s)]

    def _spawn(self, child_seed, dimension):
        """
        ABSTRACT METHOD, used by self.spawn

        Args:
            child_seeds (numpy.random.SeedSequence): length s array of seeds for each spawn
            dimension (int): lenth s array of dimensions for each spawn

        Return:
            DiscreteDistribution: spawn with new dimension using child_seed
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

class LD(DiscreteDistribution): pass

class IID(DiscreteDistribution): pass