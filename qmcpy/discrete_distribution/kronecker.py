from ._discrete_distribution import LD
from numpy import *
import time

class Kronecker(LD):
    def __init__(self, dimension=1, replications=1, randomize=False, alpha = 0, delta = 0, seed_alpha=None, seed = None, order='natural', d_max=None, m_max=None):
        # attributes required for cub_qmc_clt.py
        self.mimics = 'StdUniform'
        self.d = dimension
        self.replications = replications
        self.randomize = randomize
        self.dimension = dimension
        self.low_discrepancy = True
        self.d_max = dimension
        self.m_max = int(1e7)
        # self.order = order
        
        if sum(alpha) == 0:
            self.alpha = random.rand(dimension)
        else:
            self.alpha = alpha
        if sum(delta) == 0 and seed == None:
            self.delta = zeros(dimension)
        elif sum(delta) == 0 and seed != None:
            self.delta = random.rand(dimension)
        elif sum(delta) != 0:
            self.delta = delta

        super(Kronecker,self).__init__(dimension,seed)


    def _spawn(self, child_seed, dimension):
        return Kronecker(
                dimension=dimension,
                randomize=self.randomize,
                # order=self.order,
                seed=child_seed,
                d_max=self.d_max,
                m_max=self.m_max,
                replications=self.replications)
    

    def gen_samples(self, n=None, n_min=0, n_max=0):
        if n is None:
            n = n_max - n_min

        i = arange(n).reshape((n, 1))

        if self.randomize:
            # different for each component
            delta = random.rand(1, self.dimension)
        else:
            delta = self.delta

        return ((i * self.alpha) + delta) % 1
    

    def periodic_discrepancy(self, n, k_tilde=None, gamma=None):
        """
        Calculates the discrepancy for a periodic kernel.

        Args:
            n (int): the number of sample points
            k_tilde (tuple(function, float)): the function takes in 2 arguments: the sample points and the coordinate weights.
                The float is the integral over the unit hypercube.
            gamme (ndarray): shape (1xd)

        Returns:
            float
        
        Note:
            If k_tilde is not specified, the second Bernoulli polynomial is used.
            If gamma is not specified, the coordinate weights will be just all ones.
        """
        if gamma is None:
            gamma = ones(self.dimension)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: prod(1 + (x * (x - 1) + 1/6) * gamma, axis=1), 1)

        return sqrt(self._square_periodic_discrepancies(n, k_tilde, gamma))
        

    # calculates the weighted sum of square discrepancy
    def wssd_discrepancy(self, n, weights, k_tilde, gamma, int_k_tilde):
        discrepancies = self._square_periodic_discrepancies(n, weights, k_tilde, gamma, int_k_tilde)
        return cumsum(weights * discrepancies)
    

    def _square_periodic_discrepancies(self, n, k_tilde, gamma):
        n_array = arange(1, n + 1)
        k_tilde_terms = k_tilde[0](self.gen_samples(n=n), gamma)

        left_sum = cumsum(k_tilde_terms[1:]) * n_array[1:]
        right_sum = cumsum(n_array[:-1] * k_tilde_terms[1:])
        
        k_tilde_zero_terms = k_tilde_terms[0] * n_array
        summation = zeros(n)
        summation[1:] = left_sum - right_sum
        return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - k_tilde[1]