from numpy import *
from ._discrete_distribution import LD
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

###Start off with a code with Kronecker sequence with
###P = {x_i = i \alpha + \delta mod 1} with \alpha and \delta \in [0,1)^d

def kronecker(d, alpha = None, delta = None):
    """
    Args:
        n (int): number of samples to generate.
        d (int): number of dimensions for those desired sample points
        alpha (array): a 1 by d dimensional array in accordance to the Kronecker sequence
        as its multiplier.
            If alpha is not chosen by user, it will generate an array at which
            alpha is in [0,1)^d
            If alpha is chosen by user, it will use that vector in accordance to
            Kronecker sequence.
        delta (array): a 1 by d dimensional array in accordance to the Kronecker sequence
        as its shift.
            If delta was not chosen by user, it will generate an array such that delta is
            in [0,1)^d
            If delta is chosen by user, it will use that vector in accordance to
            Kronecker sequence.
    Returns """
    def gen_samples(self, n):
        i = arange(n).reshape((n, 1))
        return(((i*self.alpha) + self.delta)%1)