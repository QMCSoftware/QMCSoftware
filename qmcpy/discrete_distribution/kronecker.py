from ._discrete_distribution import LD
from numpy import *
import time

class Kronecker(LD):
    def __init__(self, dimension=1, alpha = 0, delta = 0, seed_alpha=None, seed_delta = None):
        self.dimension = dimension
        if sum(alpha) == 0:
            random.seed(seed_alpha)
            self.alpha = random.rand(dimension)
        else:
            self.alpha = alpha
        if sum(delta) == 0 and seed_delta == None:
            self.delta = zeros(dimension)
        elif sum(delta) == 0 and seed_delta != None:
            random.seed(seed_delta)
            self.delta = random.rand(dimension)
        elif sum(delta) != 0:
            self.delta = delta


    def gen_samples(self, n):
        i = arange(n).reshape((n, 1))
        return(((i*self.alpha) + self.delta)%1)
    
    
    def kronecker_discrepancy(self, n, k_tilde, gamma, int_k_tilde):
        n_array = arange(1, n + 1)
        k_tilde_terms = array([k_tilde(x, gamma) for x in self.gen_samples(n)])

        left_sum = cumsum(k_tilde_terms[1:]) * n_array[1:]
        right_sum = cumsum(n_array[:-1] * k_tilde_terms[1:])
        
        zero_term = array([0])
        left_sum = append(zero_term, left_sum)
        right_sum = append(zero_term, right_sum)
        
        k_tilde_zero_terms = k_tilde_terms[0] * n_array
        summation = left_sum - right_sum
        return sqrt((k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - int_k_tilde)