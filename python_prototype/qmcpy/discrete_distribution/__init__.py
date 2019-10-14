from abc import ABC, abstractmethod

from .._util import univ_repr


class DiscreteDistribution(ABC):
    """ Discrete Distribution from which we can generate samples """

    def __init__(self, mimics):
        self.mimics = mimics

    @abstractmethod
    def gen_samples(self, j, n, m):
        """
        Generate j nxm samples from the true-distribution

        Args:
            j (int): Number of nxm matrices to generate (sample.size()[0])
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])

        Returns:
            jxnxm (numpy array)
        """
        return

    def summarize(self):
        """Print important attribute values
        """
        header_fmt = "%s (%s)"
        item_s = "%35s: %-15s"
        attrs_vals_str = ""

        attrs_vals_str += header_fmt % (type(self).__name__,
                                        "Discrete Distribution Object")
        print(attrs_vals_str)

    def __repr__(self):
        return univ_repr(self, 'Discrete Distribution')


# API
from .digital_seq import DigitalSeq
from .discrete_distributions import *
