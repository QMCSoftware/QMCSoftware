from numpy import *
import time

class Kronecker:
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
    