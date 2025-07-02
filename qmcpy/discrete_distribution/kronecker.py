from ._discrete_distribution import LD
from numpy import *
import time

class Kronecker(LD):
    def __init__(self, dimension=1, replications=1, randomize=False, alpha = 0, delta = 0, seed_alpha=None, seed = None, order='natural', d_max=None, m_max=None):
        self.mimics = 'StdUniform'
        self.d = dimension
        self.replications = replications
        self.randomize = randomize
        self.dimension = dimension
        self.low_discrepancy = True
        self.d_max = dimension
        self.m_max = int(1e7)
        self.order = order
        self.dimension = dimension
        
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
                order=self.order,
                seed=child_seed,
                d_max=self.d_max,
                m_max=self.m_max,
                replications=self.replications)
    
    def gen_samples(self, n_min=0, n_max=0, n=None):
        if n is None:
            n = n_max - n_min

        i = arange(n).reshape((n, 1))

        if self.randomize:
            # different for each component
            delta = random.rand(1)
        else:
            delta = self.delta

        return ((i* self.alpha) + delta) % 1
    
    # could default k_tilde to bernoulli
    # k_tilde could be a tuple (function, integral)
    def periodic_discrepancy(self, n, k_tilde, gamma, int_k_tilde):
        n_array = arange(1, n + 1)
        k_tilde_terms = k_tilde(self.gen_samples(n=n), gamma)

        left_sum = cumsum(k_tilde_terms[1:]) * n_array[1:]
        right_sum = cumsum(n_array[:-1] * k_tilde_terms[1:])
        
        k_tilde_zero_terms = k_tilde_terms[0] * n_array
        summation = zeros(n)
        summation[1:] = left_sum - right_sum
        return sqrt((k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - int_k_tilde)
    

    def test(self, n, weights, k_tilde, gamma, int_k_tilde):
        discrepancies = self._square_periodic_discrepancies(n, weights, k_tilde, gamma, int_k_tilde) # O(N)
        return cumsum(weights * discrepancies) # O(2N)
    

    def wssd_discrepancy(self, sample, n, weights, k_tilde, gamma, int_k_tilde):
        beta = sum(weights)
        b_hat = weights / (arange(1, n + 1) ** 2)
        b_tilde = cumsum(flip(b_hat))

        A = roll(b_tilde, 1)
        A[0] = 0

        b_tilde = A + flip(b_hat)
        b = flip(cumsum(b_tilde))

        kernelkron = sample.reshape((n, 1)) * array([k_tilde(x, gamma) for x in self.gen_samples(n)])
        wssd = -1 * beta * int_k_tilde + b[0] * kernelkron[0,:] + 2 * sum((b[1:n]).reshape((n-1, 1)) * kernelkron[1:n,:], 0)
        return wssd
    

    def _square_periodic_discrepancies(self, n, weights, k_tilde, gamma, int_k_tilde):
        n_array = arange(1, n + 1)
        k_tilde_terms = array([k_tilde(x, gamma) for x in self.gen_samples(n)])

        left_sum = cumsum(k_tilde_terms[1:]) * n_array[1:]
        right_sum = cumsum(n_array[:-1] * k_tilde_terms[1:])
        
        zero_term = array([0])
        left_sum = append(zero_term, left_sum)
        right_sum = append(zero_term, right_sum)
        
        k_tilde_zero_terms = k_tilde_terms[0] * n_array
        summation = left_sum - right_sum

        return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - int_k_tilde