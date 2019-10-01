from abc import ABC, abstractmethod

from .. import univ_repr,DistributionCompatibilityError
from algorithms.distribution import DiscreteDistribution

class StoppingCriterion(ABC):
    """ Decide when to stop """
    
    def __init__(self,distrib_obj: DiscreteDistribution, allowed_distribs, abs_tol, rel_tol, n_init, n_max):
        """
        Args:
            distrib_obj: an instance of DiscreteDistribution
            allowed_distribs: distribution's compatible with the stopping criterion
            abs_tol: absolute error tolerance
            rel_tol: relative error tolerance
            n_init: initial number of samples
            n_max: maximum number of samples
        """
        if type(distrib_obj).__name__ not in allowed_distribs:
                raise DistributionCompatibilityError(type(self).__name__+' only accepts distributions:'+str(allowed_distribs))
        super().__init__()
        self.abs_tol = abs_tol if abs_tol else 1e-2 # absolute tolerance, $ d$
        self.rel_tol = rel_tol if rel_tol else 0 # relative tolerance, $ d$
        self.n_init = n_init if n_init else 1024 # initial sample size
        self.n_max = n_max if n_max else 1e8 # maximum number of samples allowed
        
    @abstractmethod
    def stopYet(self, distrib_obj: DiscreteDistribution): # distrib_obj = data or summary of data computed already
        """
        Determine when to stop

        Args:
            distrib_obj: an instance of DiscreteDistribution

        Returns:
            None

        """
        pass
    
    def __repr__(self): return univ_repr(self)