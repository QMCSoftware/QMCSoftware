from ..util import MethodImplementationError, _univ_repr, DimensionError, ParameterError
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from numpy import *


class TrueMeasure(object):
    """ True Measure abstract class. DO NOT INSTANTIATE. """

    def __init__(self):
        prefix = 'A concrete implementation of TrueMeasure must have '
        if not hasattr(self,'domain'):
            raise ParameterError(prefix + 'self.domain, 2xd ndarray of domain lower bounds (first col) and upper bounds (second col)')
        if not hasattr(self,'range'):
            raise ParameterError(prefix + 'self.range, 2xd ndarray of range lower bounds (first col) and upper bounds (second col)')
        if not hasattr(self,'parameters'):
            self.parameters = []
    
    def _parse_sampler(self, sampler):
        if isinstance(sampler,DiscreteDistribution):
            self.transform = self # this is the initial transformation, \Psi_0
            self.d = sampler.d # take the dimension from the discrete distribution
            self.discrete_distrib = sampler
            if sampler.mimics == 'StdUniform':
                if not (self.domain==tile([0,1],(self.d,1))).all():
                    raise ParameterError("The True measure's transform should have unit-cube domain.")
            elif sampler.mimics == 'StdGaussian':
                if not (self.domain==tile([-inf,inf],(self.d,1))).all():
                    raise ParameterError("The True measure's transform should have R^d domain, [-inf,inf]^d.")
            else:
                raise ParameterError('The %s true measure does not support discrete distributions that mimic a %s.'%\
                    (type(self).__name__,sampler.mimics))
        elif isinstance(sampler,TrueMeasure):
            self.transform = sampler # this is a composed transform, \Psi_j for j>0
            self.parameters += ['transform']
            self.d = sampler.d # take the dimension from the sub-sampler (composed transform)
            self.discrete_distrib = self.transform.discrete_distrib
            if self.transform.transform!=self.transform and (self.transform.domain!=self.transform.transform.range).any():
                raise ParameterError("This true measures domain must match the sub-sampling true-measures range.")
        else:
            raise ParameterError("sampler input should either be a DiscreteDistribution or TrueMeasure")
    
    def gen_samples(self, *args, **kwargs):
        """
        Generate samples from the discrete distribution
        and transform them via the transform method. 

        Args:
            args (tuple): positional arguments to the discrete distributions gen_samples method
            kwargs (dict): keyword arguments to the discrete distributions gen_samples method
        
        Returns: 
            ndarray: n x d matrix of transformed samples
        """
        x = self.discrete_distrib.gen_samples(*args,**kwargs)
        return self._transform_r(x)

    def _transform_r(self, x):
        """
        Complete transformation (recursive).
        Takes into account composed transforms.  

        Args:
            x: n x d matrix of samples mimicking a standard uniform.

        Returns:
            ndarray: n x d matrix of transformed x.  
        """
        if self.transform == self: # is \Psi_0
            return self._transform(x)
        else: # is transform \Psi_j for j>0
            xtf = self.transform._transform_r(x)
            return self._transform(xtf)
            
    def _transform(self, x): 
        """ 
        Transformation for this true measure. 

        Args:
            x: n x d matrix of samples mimicking a standard uniform.

        Returns:
            ndarray: n x d matrix of transformed x.  
        """
        raise MethodImplementationError(self,'_transform. Try setting sampler to be in a PDF TrueMeasure to importance sample by.')
        
    def _jacobian_transform_r(self, x):
        """
        Find the complete Jacobian and completely transformed samples (recursive). 
        Takes into account composed transforms. 

        Args:
            x (ndarray): n x d matrix of samples
        
        Returns:
            ndarray: length n vector of transformed samples at locations of x
            ndarray: length n vector of Jacobian values at locations of x
        """
        if self.transform == self: # is \Psi_0
            return self._transform(x),self._jacobian(x)
        else: # is transform \Psi_j for j>0
            xtf,jtf = self.transform._jacobian_transform_r(x)
            return self._transform(xtf),self._jacobian(xtf)*jtf
    
    def _jacobian(self, x):
        """
        ABSTRACT method to evaluate the Jacobian for this true measure.

        Args:
            x (ndarray): n x d matrix of samples
        
        Returns:
            ndarray: length n vector of Jacobian values at locations of x
        """
        raise MethodImplementationError(self,'jacobian. Try setting sampler to be in a PDF TrueMeasure to importance sample by.')

    def _weight(self, x):
        """
        Non-negative weight function, lambda. 
        This is often a PDF, but is not required to be 
        i.e. Lebesgue weight is always 1, but is not a PDF.

        Args:
            x (ndarray): n x d  matrix of samples
        
        Returns:
            ndarray: length n vector of weights at locations of x
        """ 
        raise MethodImplementationError(self,'weight. Try a different true measure with a weight method.')  
    
    def spawn(self, s=1, dimensions=None):
        """
        Spawn new instances of the current discrete distribution but with new seeds and dimensions. 
        Developed for multi-level and multi-replication (Q)MC algorithms.
        
        Args:
            s (int): number of spawn
            dimensions (ndarray): length s array of dimension for each spawn. Defaults to current dimension
        
        Return: 
            list: list of DiscreteDistribution instances with new seeds and dimensions
        """
        if (isinstance(dimensions,list) or isinstance(dimensions,ndarray)) and len(dimensions)==s:
            dimensions = array(dimensions)
        elif isscalar(dimensions) and dimensions%1==0:
            dimensions = tile(dimensions,s)
        elif dimensions is None:
            dimensions = tile(self.d,s)
        else:
            raise ParameterError("invalid spawn dimensions, must be None, int, or length s ndarray")
        spawns = [None]*s
        sampler_spawns = [None]*s
        for i in range(s):
            sampler = self.discrete_distrib if self.transform==self else self.transform
            sampler_spawns[i] = sampler.spawn(s=1,dimensions=[dimensions[i]])[0]
            spawns[i] = self._spawn(sampler_spawns[i],dimensions[i])
        return spawns
    
    def _spawn(self, sampler, dimension):
        """
        ABSTRACT METHOD, used by self.spawn
        
        Args:
            sampler (DiscreteDistribution or TrueMeasure): spawn of sampler for this measure
            dimension (ndarray): dimension of new spawn
        
        Return: 
            list: list of DiscreteDistribution instances with new seeds and dimensions
        """
        raise MethodImplementationError(self, '_spawn')  
        
    def __repr__(self):
        return _univ_repr(self, "TrueMeasure", self.parameters)
