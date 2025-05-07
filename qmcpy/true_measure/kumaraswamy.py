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
        array([[0.46202015, 0.5230962 ],
               [0.0352327 , 0.14904135],
               [0.26440066, 0.27701523],
               [0.13123721, 0.67157846]])
        >>> true_measure
        Kumaraswamy (AbstractTrueMeasure)
            a               [1 2]
            b               [3 4]
        
        With independent replications 

        >>> x = Kumaraswamy(DigitalNetB2(3,seed=7,replications=2),a=[1,2,3],b=[3,4,5])(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.19424882, 0.64784747, 0.69455517],
                [0.33977962, 0.30153693, 0.42157161],
                [0.06709865, 0.1267107 , 0.271294  ],
                [0.75821367, 0.47481915, 0.54237922]],
        <BLANKLINE>
               [[0.09004177, 0.22144305, 0.62190133],
                [0.47263986, 0.41449413, 0.32349266],
                [0.19457333, 0.67062492, 0.47212812],
                [0.27722181, 0.26725205, 0.80825972]]])
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
