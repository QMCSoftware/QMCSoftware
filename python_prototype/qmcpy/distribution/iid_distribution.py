""" Definition of IIDDistribution, a concrete implementation of DiscreteDistribution """

from numpy import random

from . import DiscreteDistribution


class IIDDistribution(DiscreteDistribution):
    """
    Specifies and generates the components of \
    :math:`\dfrac{1}{n} \sum_{i=1}^n \delta_{\mathbf{x}_i}(\cdot)`,
    where the :math:`\mathbf{x}_i` are IIDDistribution uniform on \
    :math:`[0,1]^d` or IIDDistribution standard Gaussian
    """

    def __init__(self, true_distribution=None, distrib_data=None, seed_rng=None
                 ):
        """
        Args:
            accepted_measures (list of strings): Measure objects compatible \
                with the DiscreteDistribution
            seed_rng (int): seed for random number generator to ensure \
                reproduciblity
        """
        accepted_measures = ["StdUniform", "StdGaussian","CustomIID"]
        if seed_rng:
            random.seed(seed_rng)
        super().__init__(accepted_measures, true_distribution, distrib_data)

    def gen_distrib(self, n, m, j=1):
        """
        Generate j nxm samples from the true-distribution

        Args:
            n (int): Number of observations (sample.size()[1])
            m (int): Number of dimensions (sample.size()[2])
            j (int): NOT USED. Number of nxm matricies to generate (sample.size()[0])

        Returns:
            nxm (numpy array)
        """
        n,m = int(n),int(m) # ensure integer types
        if type(self.true_distribution).__name__ == "StdUniform":
            samples = random.rand(n, m)
        elif type(self.true_distribution).__name__ == "StdGaussian":
            samples = random.randn(n, m)
        elif type(self.true_distribution).__name__ == "CustomIID":
            samples = self.true_distribution.generator(n, m)
        return samples
