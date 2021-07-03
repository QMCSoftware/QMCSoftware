from ..util import MethodImplementationError, _univ_repr, ParameterError
from ..true_measure._true_measure import TrueMeasure
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from numpy import *


class Integrand(object):
    """ Integrand abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        prefix = 'A concrete implementation of Integrand must have '
        if not (hasattr(self, 'true_measure') and isinstance(self.true_measure,TrueMeasure)):
            raise ParameterError(prefix + 'self.true_measure, a TrueMeasure instance')
        if not (hasattr(self, 'output_dims')):
            raise ParameterError('Set self.output_dims, the number of outputs for each input sample.')
        if not hasattr(self,'parameters'):
            self.parameters = []
        if not hasattr(self,'leveltype'):
            self.leveltype = 'single'
        self.discrete_distrib = self.true_measure.discrete_distrib
        if self.true_measure.transform!=self.true_measure and \
           not (self.true_measure.range==self.true_measure.transform.range).all():
            raise ParameterError("The range of the composed transform is not compatibe with this true measure")

    def g(self, t, *args, **kwargs):
        """
        ABSTRACT METHOD for original integrand to be integrated.

        Args:
            t (ndarray): n x d array of samples to be intput into orignal integrand. 

        Return:
            ndarray: n vector of function evaluations
        """
        raise MethodImplementationError(self, 'g')
    
    def f(self, x, *args, **kwargs):
        """
        Evalute transformed integrand based on true measures and discrete distribution 
        
        Args:
            x (ndarray): n x d array of samples from a discrete distribution
            *args: other ordered args to g
            **kwargs (dict): other keyword args to g
            
        Return: 
            ndarray: length n vector of funciton evaluations
        """
        n,d = x.shape
        if self.true_measure == self.true_measure.transform:
            # jacobian*weight/pdf will cancel so f(x) = g(\Psi(x))
            xtf = self.true_measure._transform(x) # get transformed samples, equivalent to self.true_measure._transform_r(x)
            y = self.g(xtf,*args,**kwargs).reshape(n,self.output_dims)
        else: # using importance sampling --> need to compute pdf, jacobian(s), and weight explicitly
            pdf = self.discrete_distrib.pdf(x).reshape(n,1) # pdf of samples
            xtf,jacobians = self.true_measure.transform._jacobian_transform_r(x) # compute recursive transform+jacobian
            weight = self.true_measure._weight(xtf).reshape(n,1) # weight based on the true measure
            gvals = self.g(xtf,*args,**kwargs).reshape(n,self.output_dims)
            y = gvals*weight/pdf*jacobians.reshape(n,1)
        return y

    def f_periodized(self, x, ptransform='NONE', *args, **kwargs):
        """
        Periodized transformed integrand.

        Args:
            x (ndarray): n x d array of samples from a discrete distribution
            ptransform (str): periodization transform. 
            *args: other ordered args to g
            **kwargs (dict): other keyword args to g
            
        Return: 
            ndarray: length n vector of funciton evaluations
        """
        if self.discrete_distrib.mimics != 'StdUniform':
            raise ParameterError("f_periodized requires a discrete distribution that mimics a standard uniform measure.")
        ptransform = ptransform.upper()
        n,d = x.shape
        if ptransform == 'BAKER': # Baker's transform
            xp = 1 - 2 * abs(x - 1 / 2)
            w = ones(n,dtype=float)
        elif ptransform == 'C0': # C^0 transform
            xp = 3 * x ** 2 - 2 * x ** 3
            w = prod(6 * x * (1 - x), 1)  
        elif ptransform == 'C1': # C^1 transform
            xp = x ** 3 * (10 - 15 * x + 6 * x ** 2)
            w = prod(30 * x ** 2 * (1 - x) ** 2, 1)
        elif ptransform == 'C1SIN': # Sidi C^1 transform
            xp = x - sin(2 * pi * x) / (2 * pi)
            w = prod(2 * sin(pi * x) ** 2, 1)
        elif ptransform == 'C2SIN': # Sidi C^2 transform
            xp = (8 - 9 * cos(pi * x) + cos(3 * pi * x)) / 16 # psi3
            w = prod( (9 * sin(pi * x) * pi - sin(3 * pi * x) * 3 * pi) / 16 , 1) # psi3_1
        elif ptransform == 'C3SIN': # Sidi C^3 transform
            xp = (12 * pi * x - 8 * sin(2 * pi * x) + sin(4 * pi * x)) / (12 * pi) # psi4
            w = prod( (12 * pi - 8 * cos(2 * pi * x) * 2 * pi + sin(4 * pi * x) * 4 * pi) / (12 * pi), 1) # psi4_1
        elif ptransform == 'NONE':
            xp = x
            w = ones(n,dtype=float)
        else:
            raise ParameterError("The %s periodization transform is not implemented"%ptransform)
        y = self.f(xp,*args,**kwargs)*w.reshape(n,1)
        return y
        
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
