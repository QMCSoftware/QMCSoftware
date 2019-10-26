""" DiscreteDistribution is an abstract class. """

from abc import ABC, abstractmethod

from .._util import univ_repr


class DiscreteDistribution(ABC):
    """ Discrete Distribution from which we can generate samples. """

    def __init__(self, mimics):
        """
        Initialize Discrete Distributuion instance

        Args:
            mimics (str): Measure the discrete distribution attempts to mimic
        """
        self.mimics = mimics

    @abstractmethod
    def gen_dd_samples(self, r, n, d):
        """
        Generate r nxd IID Standard Gaussian samples

        Args:
            r (int): Number of nxd matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            d (int): Number of dimensions (sample.size()[2])

        Returns:
            rxnxd (numpy array)
        """
        return

    def summarize(self):
        """ Print important attribute values """
        header_fmt = "%s (%s)"
        obj_name = "Discrete Distribution Object"
        attrs_vals_str = header_fmt % (type(self).__name__, obj_name)
        print(attrs_vals_str)

    def __repr__(self):
        return univ_repr(self, "Discrete Distribution")


# API
from .digital_seq import DigitalSeq
from .distributions import *
