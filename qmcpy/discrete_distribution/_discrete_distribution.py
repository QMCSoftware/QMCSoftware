from ..util import ParameterError, MethodImplementationError, _univ_repr, DimensionError
from numpy import *


class DiscreteDistribution(object):
    """ Discrete Distribution abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        prefix = 'A concrete implementation of DiscreteDistribution must have '
        if not hasattr(self, 'mimics'):
            raise ParameterError(prefix + 'self.mimcs (measure mimiced by the distribution)')
        if not hasattr(self, 'dimension'):
            raise ParameterError(prefix + 'self.dimension')
        if not hasattr(self, 'low_discrepancy'):
            raise ParameterError(prefix + 'self.low_discrepancy')
        if not hasattr(self,'parameters'):
            self.parameters = []

    def gen_samples(self, *args):
        """
        ABSTRACT METHOD Generate samples from discrete distribution.

        Args:
            args (tuple): tuple of positional argument. See implementations for details
        Returns:
            ndarray: $n \\times d$ array of samples, where $n$ is
            number of observations and $d$ is dimension
        """
        raise MethodImplementationError(self, 'gen_samples')

    def set_dimension(self, dimension):
        """
        ABSTRACT METHOD to reset the dimension of the problem.

        Args:
            dimension (int): new dimension to reset to

        Note:
            May not be applicable to every discrete distribution (ex: CustomIIDDistribution).
        """
        raise DimensionError("Cannot reset dimension of %s object"%str(type(self).__name__))

    def set_seed(self, seed):
        """
        ABSTRACT METHOD to reset the seed of the problem.

        Args:
            seed (int): new seed for generator

        Note:
            May not be applicable to every discrete distribution (ex: InverseCDFSampling)
        """
        raise MethodImplementationError(self, 'set_seed')

    def __repr__(self):
        return _univ_repr(self, "DiscreteDistribution", self.parameters)

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
