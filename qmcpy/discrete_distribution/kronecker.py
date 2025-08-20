from .abstract_discrete_distribution import AbstractIIDDiscreteDistribution
from numpy import *
from sympy import nextprime
import time

PRIMES = array([2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41, 
                43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97, 101,
                103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
                241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
                317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
                479, 487, 491, 499, 503, 509, 521, 523, 541])

RICHTMYER = sqrt(PRIMES) % 1

class Kronecker(AbstractIIDDiscreteDistribution):
    def __init__(self, dimension=1, replications=1, randomize=False, alpha = 0, delta = 0, seed_alpha=None, seed = None, order='natural', d_max=None, m_max=None):
        # attributes required for cub_qmc_clt.py
        self.mimics = 'StdUniform'
        self.d = dimension
        self.replications = replications
        self.randomize = randomize
        self.dimension = dimension
        self.low_discrepancy = True
        self.d_max = dimension
        self.m_max = int(1e10)
        # self.order = order
        
        # plain string
        if type(alpha) == list and type(alpha[0]) == str:
            if alpha[0].lower() == 'richtmyer':
                if dimension <= len(PRIMES):
                    self.alpha = RICHTMYER[:dimension]
                else:
                    print(len(RICHTMYER))
                    self.alpha = append(RICHTMYER, [sqrt(nextprime(PRIMES[-1], ith=x)) % 1 for x in range(1, dimension - len(PRIMES) + 1)])
        else:
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
        if gamma is None:
            gamma = ones(self.dimension)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: prod(1 + (x * (x - 1) + 1/6) * gamma, axis=1), 1)

        return sqrt(self._square_periodic_discrepancies(n, k_tilde, gamma))
       
    def _square_periodic_discrepancies(self, n, k_tilde, gamma):
        n_array = arange(1, n + 1)
        k_tilde_terms = k_tilde[0](self.gen_samples(n=n), gamma)

        left_sum = cumsum(k_tilde_terms[1:]) * n_array[1:]
        right_sum = cumsum(n_array[:-1] * k_tilde_terms[1:])
        
        k_tilde_zero_terms = k_tilde_terms[0] * n_array
        summation = zeros(n)
        summation[1:] = left_sum - right_sum
        return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - k_tilde[1]