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
        array([[1.44324713, 2.7873875 ],
               [0.32691107, 1.5741214 ],
               [1.97352511, 0.58590959],
               [0.8591331 , 1.89690854]])
        >>> true_measure
        Uniform (AbstractTrueMeasure)
            lower_bound     [0.  0.5]
            upper_bound     [2 3]

        With independent replications 

        >>> x = Uniform(DigitalNetB2(3,seed=7,replications=2),lower_bound=[.25,.5,.75],upper_bound=[1.75,1.5,1.25])(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.61979915, 0.6821862 , 1.12366296],
                [1.27229355, 1.16169442, 0.9644598 ],
                [0.97209782, 1.29818233, 0.79100643],
                [1.62311988, 0.79520621, 1.13747905]],
        <BLANKLINE>
               [[0.92315337, 1.35899604, 1.0027484 ],
                [1.05453886, 0.54353443, 0.91782473],
                [0.59821215, 0.79281506, 0.78420518],
                [1.37943573, 1.10241448, 1.13481488]]])
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
    
