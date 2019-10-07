""" Definition for abstract class DiscreteDistribution """

from abc import ABC, abstractmethod

from .._util import MeasureCompatibilityError,univ_repr

class DiscreteDistribution(ABC):
    """
    Specifies and generates the components of :math:`a_n \\sum_{i=1}^n w_i
    \\delta_{\\mathbf{x}_i}(\\cdot)`.

    Args:
        accepted_measures: list of valid measure names
        true_distribution (DiscreteDistribution): True distribution.
        distrib_data:
    Attributes:
        distribution_list: List of DiscreteDistribution instances.
        true_distribution (DiscreteDistribution): True distribution.
    Raises:
        MeasureCompatibilityError: if ``true_distribution`` not in \
        ``accepted_measures``
    """

    def __init__(self, accepted_measures, true_distribution=None,
                 distrib_data=None):
        super().__init__()
        if not true_distribution: return
        self.true_distribution = true_distribution
        if type(self.true_distribution).__name__ not in accepted_measures:
            error_message = type(
                self).__name__ + ' only accepts measures:' + str(
                accepted_measures)
            raise MeasureCompatibilityError(error_message)
        self.distribution_list = [type(self)() for i in
                                  range(len(self.true_distribution))]
        for i in range(len(self)):
            self[i].true_distribution = self.true_distribution[i]
            self[i].distrib_data = distrib_data[i] if distrib_data else None

    @abstractmethod
    def gen_distrib(self, n, m):
        """
        Generates distribution.

        Args:
            n (int): Value of :math:`n` used to determine :math:`a_n`.
            m (int):
            j (int):
        """

    def __len__(self): return len(self.distribution_list)
    def __iter__(self):
        for distribObj in self.distribution_list: yield distribObj
    def __getitem__(self, i): return self.distribution_list[i]
    def __setitem__(self, i, val): self.distribution_list[i] = val
    def __repr__(self): return univ_repr(self, 'distribution_list')

# API
from .iid_distribution import IIDDistribution
from .quasi_random import QuasiRandom