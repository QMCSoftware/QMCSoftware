from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from typing import Union

class BernoulliCont(AbstractTrueMeasure):
    r"""
    Continuous Bernoulli distribution with independent marginals as described in [https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution](https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution).

    Examples: 
        >>> true_measure = BernoulliCont(DigitalNetB2(2,seed=7),lam=.2)
        >>> true_measure(4)
        array([[0.72351141, 0.56205914],
               [0.05741849, 0.04805839],
               [0.43318125, 0.16552571],
               [0.21547493, 0.82617132]])
        >>> true_measure
        BernoulliCont (AbstractTrueMeasure)
            lam             0.200
        
        With independent replications 

        >>> x = BernoulliCont(DigitalNetB2(3,seed=7,replications=2),lam=[.25,.5,.75])(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.34826347, 0.88660568, 0.91748784],
                [0.58619073, 0.31703292, 0.45309639],
                [0.12195573, 0.0626922 , 0.15975868],
                [0.97462459, 0.64009274, 0.7015539 ]],
        <BLANKLINE>
               [[0.16343492, 0.1821862 , 0.83209443],
                [0.76587019, 0.52953251, 0.25020451],
                [0.34882573, 0.90831912, 0.56144083],
                [0.48793028, 0.25651801, 0.98567543]]])
    """

    def __init__(self, sampler, lam=1/2):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            lam (Union[float,np.ndarray]): Vector of shape parameters, each in $(0,1)$.
        """
        self.parameters = ['lam']
        self.domain = np.array([[0,1]])
        self.range = np.array([[0,1]])
        self._parse_sampler(sampler)
        self.lam = lam
        self.l = np.array(lam)
        if self.l.size==1:
            self.l = self.l.item()*np.ones(self.d)
        if not (self.l.shape==(self.d,) and (0<=self.l).all() and (self.l<=1).all()):
            raise DimensionError('lam must be scalar or have length equal to dimension and must be in (0,1).')
        super(BernoulliCont,self).__init__()

    def _transform(self, x):
        tf = np.zeros(x.shape,dtype=float)
        for j in range(self.d):
            tf[...,j] = x[...,j] if self.l[j]==1/2 else np.log(((2*self.l[j]-1)*x[...,j]-self.l[j]+1)/(1-self.l[j])) / np.log(self.l[j]/(1-self.l[j]))
        return tf
    
    def _weight(self, x):
        w = np.zeros(x.shape,dtype=float)
        for j in range(self.d):
            C = 2 if self.l[j]==1/2 else 2*np.arctanh(1-2*self.l[j])/(1-2*self.l[j])
            w[...,j] = C*self.l[j]**x[...,j]*(1-self.l[j])**(1-x[...,j])
        return np.prod(w,-1)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = BernoulliCont(sampler,lam=self.lam)
        else:
            l0 = self.l[0]
            if (self.l!=l0).any():
                raise DimensionError('''
                        In order to spawn a BernoulliCont measure
                        lam must all be the same.''')
            spawn = BernoulliCont(sampler,lam=l0)
        return spawn
