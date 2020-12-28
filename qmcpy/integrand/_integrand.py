from ..util import MethodImplementationError, TransformError, _univ_repr, ParameterError
from numpy import *


class Integrand(object):
    """ Integrand abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        prefix = 'A concrete implementation of Integrand must have '
        if not hasattr(self, 'measure'):
            raise ParameterError(prefix + 'self.measure (a TrueMeasure instance)')
        if not hasattr(self, 'distribution'):
            raise ParameterError(prefix + 'self.distribution (a DiscreteDistribuiton instance')
        if not hasattr(self,'parameters'):
            self.parameters = []
        self.dimension = self.measure.dimension
        if not hasattr(self,'leveltype'):
            self.leveltype = 'single'
    
    def f(self, x, *args, **kwargs):
        """ transformed integrand. 
        
        Args:
            x (ndarray): n x d array of samples from a discrete distribution
            *args: ordered args to g
            **kwargs (dict): keyword args to g
            
        Return: 
            ndarray: length n vector of funciton evaluations
        """
        return self.measure._eval_f(x,self.g,*args,**kwargs)
        
    def period_transform(self, ptransform):
        """ Computes the periodization transform for the given function values """
        if self.distribution.mimics != 'StdUniform':
            raise Exception("period_transform requires discrete distribution to mimic the standard uniform")
        if ptransform == 'Baker':
            f = lambda x: self.f(1 - 2 * abs(x - 1 / 2))  # Baker's transform
        elif ptransform == 'C0':
            f = lambda x: self.f(3 * x ** 2 - 2 * x ** 3) * prod(6 * x * (1 - x), 1)  # C^0 transform
        elif ptransform == 'C1':
            # C^1 transform
            f = lambda x: self.f(x ** 3 * (10 - 15 * x + 6 * x ** 2)) * prod(30 * x ** 2 * (1 - x) ** 2, 1)
        elif ptransform == 'C1sin':
            # Sidi C^1 transform
            f = lambda x: self.f(x - sin(2 * pi * x) / (2 * pi)) * prod(2 * sin(pi * x) ** 2, 1)
        elif ptransform == 'C2sin':
            # Sidi C^2 transform
            psi3 = lambda t: (8 - 9 * cos(pi * t) + cos(3 * pi * t)) / 16
            psi3_1 = lambda t: (9 * sin(pi * t) * pi - sin(3 * pi * t) * 3 * pi) / 16
            f = lambda x: self.f(psi3(x)) * prod(psi3_1(x), 1)
        elif ptransform == 'C3sin':
            # Sidi C^3 transform
            psi4 = lambda t: (12 * pi * t - 8 * sin(2 * pi * t) + sin(4 * pi * t)) / (12 * pi)
            psi4_1 = lambda t: (12 * pi - 8 * cos(2 * pi * t) * 2 * pi + sin(
                4 * pi * t) * 4 * pi) / (12 * pi)
            f = lambda x: self.f(psi4(x)) * prod(psi4_1(x), 1)
        elif ptransform == 'none':
            # do nothing
            f = lambda x: self.f(x)
        else:
            f = self.f
            print(f'Error: Periodization transform {ptransform} not implemented')
        return f

    def g(self, x):
        """
        ABSTRACT METHOD for original integrand to be integrated.

        Args:
            x (ndarray): n samples by d dimension array of samples 
                generated according to the true measure. 
            l (int): OPTIONAL input for multi-level integrands. The level to generate at. 
                Note that the dimension of x is determined by the _dim_at_level method for 
                multi-level methods.

        Return:
            ndarray: n vector of function evaluations
        """
        raise MethodImplementationError(self, 'g')
    
    def _dim_at_level(self, l):
        """
        ABSTRACT METHOD to return the dimension of samples to generate at level l. 
        This method only needs to be implemented for multi-level integrands where 
        the dimension changes depending on the level. 
        
        Args:
            l (int): level
        
        Return:
            int: dimension of samples needed at level l
        """
        raise MethodImplementationError(self, '_dim_at_level')

    def __repr__(self):
        return _univ_repr(self, "Integrand", self.parameters)

    def plot(self, *args, **kwargs):
        """ Create a plot relevant to the true measure object. """
        raise MethodImplementationError(self,'plot')
