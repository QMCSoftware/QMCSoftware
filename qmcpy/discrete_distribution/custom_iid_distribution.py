""" Definition for CustomIIDDistribution, a concrete implementation of DiscreteDistribution """

from ._discrete_distribution import DiscreteDistribution


class CustomIIDDistribution(DiscreteDistribution):
    """ Wrapper around user's discrete distribution generator """

    parameters = []

    def __init__(self, custom_generator):
        """
        Args:
            custom_generator (function): custom generator of discrete distribution
        """
        self.custom_generator = custom_generator
        self.distrib_type = 'iid'
        self.mimics = 'Custom'
        self.dimension = None
        super().__init__()

    def gen_samples(self, n):
        """
        Generate n iid samples

        Args:
            n (int): Number of observations to generate

        Returns:
            n iid samples from CustomDistribution
        """
        return self.custom_generator(n)

