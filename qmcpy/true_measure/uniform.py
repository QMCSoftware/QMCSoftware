from ._true_measure import TrueMeasure
from ..util import TransformError, DimensionError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from scipy.stats import norm


class Uniform(TrueMeasure):
    """
    >>> u = Uniform(DigitalNetB2(2,seed=7),lower_bound=[0,.5],upper_bound=[2,3])
    >>> u.gen_samples(4)
    array([[0.1431124 , 0.69460269],
           [1.62840337, 2.36213894],
           [0.62818597, 2.83084783],
           [1.14326115, 1.16339383]])
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
        self.domain = np.array([[0,1]])
        self._parse_sampler(sampler)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if np.isscalar(self.lower_bound):
            lower_bound = np.tile(self.lower_bound,self.d)
        if np.isscalar(self.upper_bound):
            upper_bound = np.tile(self.upper_bound,self.d)
        self.a = np.array(lower_bound)
        self.b = np.array(upper_bound)
        if len(self.a)!=self.d or len(self.b)!=self.d:
            raise DimensionError('upper bound and lower bound must be of length dimension')
        self._set_constants()
        self.range = np.hstack((self.a.reshape((self.d,1)),self.b.reshape((self.d,1))))
        super(Uniform,self).__init__()

    def _set_constants(self):
        self.delta = self.b - self.a
        self.inv_delta_prod = 1/self.delta.prod()

    def _transform(self, x):
        return x * self.delta + self.a
    
    def _weight(self, x):
        return np.tile(self.inv_delta_prod,x.shape[0])
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = Uniform(sampler,lower_bound=self.a,upper_bound=self.b)
        else:
            l = self.a[0]
            u = self.b[0]
            if not (all(self.a==l) and all(self.b==u)):
                raise DimensionError('''
                    In order to spawn a uniform measure
                    the lower bounds must all be the same and 
                    the upper bounds must all be the same''')
            spawn = Uniform(sampler,lower_bound=l,upper_bound=u)
        return spawn
    
