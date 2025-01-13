from .._discrete_distribution import LD
from ...util import ParameterError, ParameterWarning
from numpy import *
from os.path import dirname, abspath, isfile
import warnings

###For kronecker sequence let the user punch in alpha
###Look at the paper on overleaf for Kronecker sequence
###Try and find a way to get the sequence accurately
###For now get a random generators for alpha
###Step 2 investigate the median

###Start off with a code with Kronecker sequence with
###P = {x_i = i \alpha + \delta mod 1} with \alpha and \delta \in [0,1)^d
def kronecker(n, d, alpha = 1, delta = 1):
    i = arange(1, n+1).reshape((n, 1))
    if alpha == 1:
        alpha = random.rand(d)
    if delta == 1:
        delta = random.rand(d)
    return(((i*alpha) + delta)%1)