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
    def g(self, x):
        """
        Original integrand to be integrated

        Args:
            x: nodes, :math:`\mathbf{x}_{\mathfrak{u},i} = i^{\mathtt{th}}` \
                row of an :math:`n \cdot |\mathfrak{u}|` matrix

        Returns:
            :math:`n \cdot p` matrix with values \
            :math:`f(\mathbf{x}_{\mathfrak{u},i},\mathbf{c})` where if \
            :math:`\mathbf{x}_i' = (x_{i, \mathfrak{u}},\mathbf{c})_j`, then \
            :math:`x'_{ij} = x_{ij}` for :math:`j \in \mathfrak{u}`, and \
            :math:`x'_{ij} = c` otherwise
        """
        pass

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
