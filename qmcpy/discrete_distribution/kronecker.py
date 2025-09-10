from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from numpy import *
from sympy import nextprime

PRIMES = array([2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41, 
                43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97, 101,
                103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
                241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
                317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
                479, 487, 491, 499, 503, 509, 521, 523, 541])

RICHTMYER = sqrt(PRIMES) % 1
SUZUKI = lambda d : 2**(arange(1,d+1)/(d+1))

class Kronecker(AbstractLDDiscreteDistribution):
    def __init__(self, dimension=1, alpha="RICHTMYER", delta=None, replications=None, randomize=True, seed=None):
        # attributes required for cub_qmc_clt.py
        self.mimics = 'StdUniform'
        self.randomize = randomize
        
        if type(alpha) == str and alpha.lower() == 'richtmyer':
                if dimension <= len(PRIMES):
                    self.alpha = RICHTMYER[:dimension]
                else:
                    self.alpha = append(RICHTMYER, [sqrt(nextprime(PRIMES[-1], ith=x)) % 1 for x in range(1, dimension - len(PRIMES) + 1)])
        elif type(alpha) == str and alpha.lower() == 'suzuki':
            self.alpha = SUZUKI(dimension)
        else:
            self.alpha = alpha[:dimension]

        super(Kronecker,self).__init__(dimension,replications,seed,d_limit=dimension,n_limit=inf) 

        if self.randomize:
            self.delta = self.rng.uniform(size =(self.replications, self.d))
        elif delta is not None:
            self.delta = delta * ones((self.replications, self.d))
        else:
            self.delta = zeros((self.replications, self.d))

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        # returns replications x (n_max-n_min) x d (dimension) array of samples

        i = arange(n_min,n_max).reshape((n_max-n_min, 1))

        points = ((i * self.alpha) + self.delta[:,None,:]) % 1

        return points


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
            gamma = ones(self.d)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: prod(1 + (x * (x - 1) + 1/6) * gamma, axis=-1), 1)

        return sqrt(self._square_periodic_discrepancies(n, k_tilde, gamma))
        

    # calculates the weighted sum of square discrepancy
    def wssd_discrepancy(self, n, weights, k_tilde = None, gamma = None):
        if gamma is None:
            gamma = ones(self.d)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: prod(1 + (x * (x - 1) + 1/6) * gamma, axis=-1), 1)

        discrepancies = self._square_periodic_discrepancies(n, k_tilde, gamma)
        return sum(weights * discrepancies, axis=-1)
    
    
    def _square_periodic_discrepancies(self, n, k_tilde, gamma):
        n_array = arange(1, n + 1)
        k_tilde_terms = k_tilde[0](self.gen_samples(n=n), gamma)

        left_sum = cumsum(k_tilde_terms[...,1:], axis=-1) * n_array[1:]
        right_sum = cumsum(n_array[:-1] * k_tilde_terms[...,1:], axis=-1)

        k_tilde_zero_terms = k_tilde_terms[...,0] * n_array
        summation = zeros_like(k_tilde_terms)
        summation[...,1:] = left_sum - right_sum
        return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - k_tilde[1]
    
    
    def _spawn(self, child_seed, dimension):
        return Kronecker(
                dimension=dimension,
                alpha = self.alpha,
                randomize=self.randomize,
                seed=child_seed,
                d_max=self.d_max,
                m_max=self.m_max,
                replications=self.replications)