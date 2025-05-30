from numpy import *
import time

###For kronecker sequence let the user punch in alpha
###Look at the paper on overleaf for Kronecker sequence
###Try and find a way to get the sequence accurately
###For now get a random generators for alpha
###Step 2 investigate the median

###Start off with a code with Kronecker sequence with
###P = {x_i = i \alpha + \delta mod 1} with \alpha and \delta \in [0,1)^d
def kronecker(n, d, alpha = None, delta = None):
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

        Note:
            n and d are required in order to get an output of the list of samples x_1 to x_n
    """
    i = arange(n).reshape((n, 1))
    #line 33 gives out a list of natural numbers ranging from 1 to integer variable "n" given by the user. 
    if any(x is None for x in alpha):           #if alpha is not chosen by user, it will choose a randomly generated vector in [0,1)^d.
        alpha = random.rand(d)
    if delta == None:           #if delta is not chosen by user, it will choose a randomly generated vector in [0,1)^d.
        delta = random.rand(d)
    return(((i*alpha) + delta)%1)   #in order to find the Kronecker sequence take integer i, multiply it by alpha, then
                                    #take modular 1, so that the vector is in [0,1)^d.