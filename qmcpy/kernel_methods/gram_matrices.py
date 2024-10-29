from ..discrete_distribution import Lattice
from .shift_invar_ops import KernelShiftInvar
from .dig_shift_invar_ops import DigitallyShiftInvariantKernel
import numpy as np 
    
class FastGramMatrix(object):
    def __init__(self):
        pass

class FastGramMatrixLattice(object):
    def __init__(self, lattice_obj, si_kernel, n):
        assert isinstance(lattice_obj,Lattice)
        assert isinstance(si_kernel,ShiftInvariantKernel)
        assert (n&(n-1))==0 # require n is 0 or a power of 2 
        self.x = lattice_obj.gen_samples(n_min=0,n_max=n)




