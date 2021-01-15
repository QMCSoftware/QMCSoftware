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

    def _set_dimension(self, dimension):
        """
        ABSTRACT METHOD to reset the dimension for this true measure. 

        Args:
            dimension (int): new dimension to reset to 
        """
        raise DimensionError("Cannot reset dimension of %s object"%str(type(self).__name__))
    
    def _set_dimension_r(self, dimension):
        """
        Completely reset the dimension. 
        Takes into account composed transforms. 
        """
        self._set_dimension(dimension)
        if self.transform == self: # is \Psi_0
            self.discrete_distrib._set_dimension(dimension)
        else: # is \Psi_j for some j>0
            self.transform._set_dimension_r(dimension)
    
    def _weight(self, x):
        """
        Non-negative weight function, \lambda. 
        This is often a PDF, but is not required to be 
        i.e. Lebesgue weight is always 1, but is not a PDF.

        Args:
            x (ndarray): n x d  matrix of samples
        
        Returns:
            ndarray: length n vector of weights at locations of x
        """ 
        raise MethodImplementationError(self,'weight. Try a different true measure with a weight method.')

    def _parse_sampler(self, sampler):
        """
        Parse the sampler input to any TrueMeasure instance.

        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform. 
        """
        if isinstance(sampler,DiscreteDistribution):
            self.transform = self # this is the initial transformation, \Psi_0
            self.d = sampler.d # take the dimension from the discrete distribution
            self.discrete_distrib = sampler
            if sampler.mimics == 'StdUniform':
                if not (self.domain==tile([0,1],(self.d,1))).all():
                    raise ParameterError("The True measure's transform should have unit-cube domain.")
            else:
                raise ParameterError("True measures currently only support discrete distributions that mimic the standard uniform")
        elif isinstance(sampler,TrueMeasure):
            self.transform = sampler # this is a composed transform, \Psi_j for j>0
            self.parameters += ['transform']
            self.d = sampler.d # take the dimension from the sub-sampler (composed transform)
            self.discrete_distrib = self.transform.discrete_distrib
            if self.transform.transform!=self.transform and (self.domain!=self.transform.range).any():
                raise ParameterError("This true measures domain must match the sub-sampling true-measures range.")
        else:
            raise ParameterError("sampler input should either be a DiscreteDistribution or TrueMeasure")
        
    def __repr__(self):
        return _univ_repr(self, "TrueMeasure", self.parameters)
