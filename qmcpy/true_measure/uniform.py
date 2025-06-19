from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from scipy.stats import norm
from typing import Union


class Uniform(AbstractTrueMeasure):
    r"""
    Uniform distribution, see [https://en.wikipedia.org/wiki/Continuous_uniform_distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution). 

    Examples:
        >>> true_measure = Uniform(DigitalNetB2(2,seed=7),lower_bound=[0,.5],upper_bound=[2,3])
        >>> true_measure(4)
        array([[1.68859325, 2.30405891],
               [0.20403559, 0.71484077],
               [1.203925  , 1.18347694],
               [0.68860467, 2.77292277]])
        >>> true_measure
        Uniform (AbstractTrueMeasure)
            lower_bound     [0.  0.5]
            upper_bound     [2 3]

        With independent replications 

        >>> x = Uniform(DigitalNetB2(3,seed=7,replications=2),lower_bound=[.25,.5,.75],upper_bound=[1.75,1.5,1.25])(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.96531723, 1.38660568, 1.18500377],
                [1.31832387, 0.81703292, 0.91126517],
                [0.53213705, 0.5626922 , 0.79796433],
                [1.72879753, 1.14009274, 1.04033896]],
        <BLANKLINE>
               [[0.61979915, 0.6821862 , 1.12366296],
                [1.53000482, 1.02953251, 0.82909243],
                [0.96626492, 1.40831912, 0.96324993],
                [1.18362201, 0.75651801, 1.23828953]]])
    """
    
    def __init__(self, sampler, lower_bound=0, upper_bound=1):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            lower_bound (Union[float,np.ndarray]): Lower bound.
            upper_bound (Union[float,np.ndarray]): Upper bound.
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
        self.delta = self.b - self.a
        self.inv_delta_prod = 1/self.delta.prod()
        self.range = np.hstack((self.a.reshape((self.d,1)),self.b.reshape((self.d,1))))
        super(Uniform,self).__init__()
        assert self.a.shape==(self.d,) and self.b.shape==(self.d,)

    def _transform(self, x):
        return x*self.delta+self.a
    
    def _weight(self, x):
        return np.tile(self.inv_delta_prod,x.shape[:-1])
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = Uniform(sampler,lower_bound=self.a,upper_bound=self.b)
        else:
            l = self.a[0]
            u = self.b[0]
            if not (all(self.a==l) and all(self.b==u)):
                raise DimensionError('''
                    In order to spawn a uniform measure with different dimensions
                    the lower bounds must all be the same and 
                    the upper bounds must all be the same''')
            spawn = Uniform(sampler,lower_bound=l,upper_bound=u)
        return spawn
    
