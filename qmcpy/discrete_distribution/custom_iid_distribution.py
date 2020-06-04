from ._discrete_distribution import DiscreteDistribution


class CustomIIDDistribution(DiscreteDistribution):

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
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        return self.custom_generator(n)

