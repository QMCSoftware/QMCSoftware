""" Definition for abstract class, ``Integrand`` """

from abc import ABC, abstractmethod
from numpy import cumsum, diff, insert, sqrt
from scipy.stats import norm

from .._util import univ_repr, TransformError


class Integrand(ABC):
    def __init__(self):
        """
        Specify and generate values :math:`f(\mathbf{x})` for \
        :math:`\mathbf{x} \in \mathcal{X}`.

        Attributes:
            f (Integrand): function transformed to accept distribution \
                values
            dimension (int): dimension of the domain, :math:`d > 0`
            integrand_list (list): list of Integrands, may be more than 1 for \
                multi-dimensional problems
        """
        super().__init__()
        self.f = None
        self.dimension = 2
        self.integrand_list = [self]

    # Abstract Methods
    @abstractmethod
    def g(self, x, coord_index):
        """
        Original integrand to be integrated

        Args:
            x: nodes, :math:`\mathbf{x}_{\mathfrak{u},i} = i^{\mathtt{th}}` \
                row of an :math:`n \cdot |\mathfrak{u}|` matrix
            coord_index: set of those coordinates in sequence needed, \
                :math:`\mathfrak{u}`

        Returns:
            :math:`n \cdot p` matrix with values \
            :math:`f(\mathbf{x}_{\mathfrak{u},i},\mathbf{c})` where if \
            :math:`\mathbf{x}_i' = (x_{i, \mathfrak{u}},\mathbf{c})_j`, then \
            :math:`x'_{ij} = x_{ij}` for :math:`j \in \mathfrak{u}`, and \
            :math:`x'_{ij} = c` otherwise
        """
        pass

    def transform_variable(self, measure, distribution):
        """
        This method performs the necessary variable transformation to put the
        original integrand in the form required by the DiscreteDistribution
        object starting from the original Measure object

        Args:
            measure (Measure): the Measure object that defines the integral
            distribution (DiscreteDistribution): the discrete distribution \
                object that is sampled from

        Returns: None
        """
        for i in range(len(self)):
            try:
                sample_from = distribution[i].true_distribution.mimics
                    # QuasiRandom sampling
            except:
                sample_from = type(distribution[i].true_distribution).__name__
                    # IIDDistribution sampling
            transform_to = type(measure[i]).__name__
            # distribution the sampling attempts to mimic
            self[i].dimension = distribution[i].true_distribution.dimension
            # the integrand needs the dimension
            if transform_to == sample_from:  # no need to transform
                self[i].f = lambda xu, coordIdex, i=i: self[i].g(xu, coordIdex)
            elif transform_to == 'IIDZeroMeanGaussian' and \
                sample_from == 'StdGaussian':  # multiply by likelihood ratio
                this_var = measure[i].variance
                self[i].f = lambda xu, coordIndex, var=this_var, i=i: \
                    self[i].g(xu * sqrt(var), coordIndex)
            elif transform_to == 'IIDZeroMeanGaussian' and \
                sample_from == 'StdUniform':  # inverse cdf transform
                this_var = measure[i].variance
                self[i].f = lambda xu, coordIdex, var=this_var, i=i: \
                    self[i].g(sqrt(var) * norm.ppf(xu), coordIdex)
            elif transform_to == 'BrownianMotion' and \
                sample_from == 'StdUniform':
                # inverse cdf transform -> sum across time-series
                time_diff = diff(insert(measure[i].time_vector, 0, 0))
                self[i].f = lambda xu, coordIndex, timeDiff=time_diff, i=i: \
                    self[i].g(cumsum(norm.ppf(xu) * sqrt(timeDiff), 1),
                              coordIndex)
            elif transform_to == 'BrownianMotion' and \
                sample_from == 'StdGaussian':  # sum across time-series
                time_diff = diff(insert(measure[i].time_vector, 0, 0))
                self[i].f = lambda xu, coordIndex, timeDiff=time_diff, i=i: \
                    self[i].g(cumsum(xu * sqrt(timeDiff), 1), coordIndex)
            else:
                msg = "Cannot transform %s distribution to mimic Integrand's " \
                      "true %s measure"
                raise TransformError(msg % (sample_from, transform_to))

        return

    def __len__(self):
        return len(self.integrand_list)

    def __iter__(self):
        for fun in self.integrand_list:
            yield fun

    def __getitem__(self, i):
        return self.integrand_list[i]

    def __setitem__(self, i, val):
        self.integrand_list[i] = val

    def __repr__(self):
        return univ_repr(self, "integrand_list")

    def summarize(self):
        header_fmt = "%s (%s)"
        attrs_vals_str = header_fmt % (type(self).__name__, "Integrand Object")
        print(attrs_vals_str)


# API
from .asian_call import AsianCall
from .keister import Keister
from .linear import Linear
from .quick_construct import QuickConstruct
