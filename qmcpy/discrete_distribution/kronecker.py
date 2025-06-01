from numpy import *
import time

class Kronecker:
    def __init__(self, dimension=1, alpha = None, delta = 0, seed_alpha=None, seed_delta = None):
        self.dimension = dimension
        if alpha == None:
            random.seed(seed_alpha)
            self.alpha = random.rand(dimension)
        else:
            self.alpha = alpha
        if delta == 0 and seed_delta == None:
            self.delta = zeros(dimension)
        elif delta == 0 and seed_delta != None:
            random.seed(seed_delta)
            self.delta = random.rand(dimension)
        elif delta != 0:
            self.delta = delta


    def gen_samples(self, n):
        i = arange(n).reshape((n, 1))
        #line 20 gives out a list of natural numbers ranging from 1 to integer variable "n" given by the user. 
        return(((i*self.alpha) + self.delta)%1)   #in order to find the Kronecker sequence take integer i, multiply it by alpha, then
                                    #take modular 1, so that the vector is in [0,1)^d.