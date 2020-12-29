from ._true_measure import TrueMeasure
from ..util import TransformError, DimensionError
from ..discrete_distribution import Sobol
from numpy import *
from scipy.stats import norm


class Uniform(TrueMeasure):
    """
    >>> d = 2
    >>> u = Uniform(d,lower_bound=1,upper_bound=2)
    >>> u
    Uniform (TrueMeasure Object)
        d               2
        lower_bound     1
        upper_bound     2
    >>> s = Sobol(d, seed=7)
    >>> u.transform(s.gen_samples(n_min=4,n_max=8))
    array([[1.537, 1.917],
           [1.253, 1.2  ],
           [1.799, 1.417],
           [1.02 , 1.7  ]])
    >>> d_new = 4
    >>> u.set_dimension(d_new)
    >>> s.set_dimension(d_new)
    >>> u.transform(s.gen_samples(n_min=4,n_max=8))
    array([[1.537, 1.917, 1.78 , 1.13 ],
           [1.253, 1.2  , 1.27 , 1.798],
           [1.799, 1.417, 1.068, 1.619],
           [1.02 , 1.7  , 1.578, 1.451]])
    >>> u2 = Uniform(2, lower_bound=[-.5,0], upper_bound=[1,3])
    >>> u2
    Uniform (TrueMeasure Object)
        d               2
        lower_bound     [-.5, 0]
        upper_bound     [1, 3]
    >>> u2.pdf([[0,1]])
    array([0.222])
    """

    parameters = ['d', 'lower_bound', 'upper_bound']
    
    def __init__(self, dimension=1, lower_bound=0., upper_bound=1.):
        """
        Args:
            dimension (int): dimension of the distribution
            lower_bound (float): a for Uniform(a,b)
            upper_bound (float): b for Uniform(a,b)
        """
        self.d = self.dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if isscalar(self.lower_bound):
            a = tile(self.lower_bound,self.d)
        if isscalar(self.upper_bound):
            b = tile(self.upper_bound,self.d)
        self.a = array(a)
        self.b = array(b)
        if len(self.a)!=self.d or len(self.b)!=self.d:
            raise DimensionError('upper bound and lower bound must be of length dimension')
        self._set_constants()
        super(Uniform,self).__init__() 

    def _set_constants(self):
        self.delta = self.b - self.a
        self.delta_prod = self.delta.prod()
        self.inv_delta_prod = 1/self.delta_prod

    def transform(self, x):
        return x * self.delta + self.a

    def jacobian(self, x):
        return tile(self.delta_prod,x.shape[0])
    
    def weight(self, x):
        return tile(self.inv_delta_prod,x.shape[0])
    
    def set_dimension(self, dimension):
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
    