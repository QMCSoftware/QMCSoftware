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


    def gen_samples(self, n):
        i = arange(n).reshape((n, 1))
        #line 20 gives out a list of natural numbers ranging from 1 to integer variable "n" given by the user. 
        return(((i*self.alpha) + self.delta)%1)   #in order to find the Kronecker sequence take integer i, multiply it by alpha, then
                                    #take modular 1, so that the vector is in [0,1)^d.