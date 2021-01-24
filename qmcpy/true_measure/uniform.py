from ._true_measure import TrueMeasure
from ..util import TransformError, DimensionError
from ..discrete_distribution import Sobol
from numpy import *
from scipy.stats import norm


class Uniform(TrueMeasure):
    """
    >>> u = Uniform(Sobol(2,seed=7),lower_bound=[0,.5],upper_bound=[2,3])
    >>> u.gen_samples(4)
    array([[1.346, 0.658],
           [0.797, 2.471],
           [1.826, 1.909],
           [0.25 , 1.22 ]])
    >>> u
    Uniform (TrueMeasure Object)
        lower_bound     [0.  0.5]
        upper_bound     [2 3]
    """
    
    def __init__(self, sampler, lower_bound=0., upper_bound=1.):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        self.parameters = ['lower_bound', 'upper_bound']
        self.domain = array([[0,1]])
        self._parse_sampler(sampler)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if isscalar(self.lower_bound):
            lower_bound = tile(self.lower_bound,self.d)
        if isscalar(self.upper_bound):
            upper_bound = tile(self.upper_bound,self.d)
        self.a = array(lower_bound)
        self.b = array(upper_bound)
        if len(self.a)!=self.d or len(self.b)!=self.d:
            raise DimensionError('upper bound and lower bound must be of length dimension')
        self._set_constants()
        self.range = hstack((self.a,self.b))
        super(Uniform,self).__init__() 

    def _set_constants(self):
        self.delta = self.b - self.a
        self.delta_prod = self.delta.prod()
        self.inv_delta_prod = 1/self.delta_prod

    def _transform(self, x):
        return x * self.delta + self.a

    def _jacobian(self, x):
        return tile(self.delta_prod,x.shape[0])
    
    def _weight(self, x):
        return tile(self.inv_delta_prod,x.shape[0])
    
    def _set_dimension(self, dimension):
        l = self.a[0]
        u = self.b[0]
        if not (all(self.a==l) and all(self.b==u)):
            raise DimensionError('''
                In order to change dimension of uniform measure
                the lower bounds must all be the same and 
                the upper bounds must all be the same''')
        self.d = dimension
        self.a = tile(l,self.d)
        self.b = tile(u,self.d)
        self._set_constants()
    