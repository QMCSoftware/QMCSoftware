from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from typing import Union


class Kumaraswamy(AbstractTrueMeasure):
    r"""
    Kumaraswamy distribution as described in [https://en.wikipedia.org/wiki/Kumaraswamy_distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution).
    
    Examples:
        >>> true_measure = Kumaraswamy(DigitalNetB2(2,seed=7),a=[1,2],b=[3,4])
        >>> true_measure(4)
        array([[0.34705366, 0.6782161 ],
               [0.0577568 , 0.36189538],
               [0.76344358, 0.0932949 ],
               [0.17065545, 0.43009386]])
        >>> true_measure
        Kumaraswamy (AbstractTrueMeasure)
            a               [1 2]
            b               [3 4]
        
        With independent replications 

        >>> x = Kumaraswamy(DigitalNetB2(3,seed=7,replications=2),a=[1,2,3],b=[3,4,5])(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.09004177, 0.22144305, 0.62190133],
                [0.31710078, 0.48718217, 0.47325643],
                [0.19657641, 0.57423463, 0.25697057],
                [0.56103074, 0.28939035, 0.63654112]],
        <BLANKLINE>
               [[0.18006788, 0.62226635, 0.5083556 ],
                [0.22602452, 0.10519477, 0.42823814],
                [0.08428482, 0.28804621, 0.2414302 ],
                [0.37253319, 0.45379743, 0.63366422]]])
    """

    def __init__(self, sampler, a=2, b=2):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            a (Union[float,np.ndarray]): First parameter $\alpha > 0$.
            b (Union[float,np.ndarray]): Second parameter $\beta > 0$.
        """
        self.parameters = ['a', 'b']
        self.domain = np.array([[0,1]])
        self.range = np.array([[0,1]])
        self._parse_sampler(sampler)
        self.a = a
        self.b = b
        self.alpha = np.array(a)
        if self.alpha.size==1:
            self.alpha = self.alpha.item()*np.ones(self.d)
            a = np.tile(self.a,self.d)
        self.beta = np.array(b)
        if self.beta.size==1:
            self.beta = self.beta.item()*np.ones(self.d)
        if not (self.alpha.shape==(self.d,) and self.beta.shape==(self.d,)):
            raise DimensionError('a and b must be scalar or have length equal to dimension.')
        if not ((self.alpha>0).all() and (self.beta>0).all()):
            raise ParameterError("Kumaraswamy requires a,b>0.")
        super(Kumaraswamy,self).__init__()
        assert self.alpha.shape==(self.d,) and self.beta.shape==(self.d,)

    def _transform(self, x):
        return (1-(1-x)**(1/self.beta))**(1/self.alpha)
    
    def _weight(self, x):
        return np.prod(self.alpha*self.beta*x**(self.alpha-1)*(1-x**self.alpha)**(self.beta-1),-1)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = Kumaraswamy(sampler,a=self.alpha,b=self.beta)
        else:
            a = self.alpha[0]
            b = self.beta[0]
            if not (all(self.alpha==a) and all(self.beta==b)):
                raise DimensionError('''
                    In order to spawn a Kumaraswamy measure
                    a must all be the same and 
                    b must all be the same''')
            spawn = Kumaraswamy(sampler,a=a,b=b)
        return spawn
