from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from ..discrete_distribution import DigitalNetB2
import numpy as np
import scipy.stats
from typing import Union


class SciPyWrapper(AbstractTrueMeasure):
    r"""
    Multivariate distribution with independent marginals from [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions).

    Examples:
        >>> true_measure = SciPyWrapper(
        ...     sampler = DigitalNetB2(3,seed=7),
        ...     scipy_distribs = [
        ...         scipy.stats.uniform(loc=1,scale=2),
        ...         scipy.stats.norm(loc=3,scale=4),
        ...         scipy.stats.gamma(a=5,loc=6,scale=7)])
        >>> true_measure.range
        array([[  1.,   3.],
               [-inf,  inf],
               [  6.,  inf]])
        >>> true_measure(4)
        array([[ 2.19095365, 10.91497366, 44.06314171],
               [ 1.8834004 ,  1.94882115, 24.81117897],
               [ 2.69736377, -0.67521937, 30.12172737],
               [ 1.37601847,  5.01292061, 50.09489436]])
        
        >>> true_measure = SciPyWrapper(sampler=DigitalNetB2(2,seed=7),scipy_distribs=scipy.stats.beta(a=5,b=1))
        >>> true_measure(4)
        array([[0.96671623, 0.93683292],
               [0.63348332, 0.61211827],
               [0.90347031, 0.77153702],
               [0.80795653, 0.98113537]])
        
        With independent replications 

        >>> x = SciPyWrapper(
        ...     sampler = DigitalNetB2(3,seed=7,replications=2),
        ...     scipy_distribs = [
        ...         scipy.stats.uniform(loc=1,scale=2),
        ...         scipy.stats.norm(loc=3,scale=4),
        ...         scipy.stats.gamma(a=5,loc=6,scale=7)])(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[ 1.95375631,  7.83469056, 58.69999116],
                [ 2.42443182,  1.09595201, 32.25281658],
                [ 1.37618274, -3.13023855, 22.80335185],
                [ 2.97173004,  4.4348268 , 41.85528858]],
        <BLANKLINE>
               [[ 1.49306553, -0.62826013, 49.7677589 ],
                [ 2.7066731 ,  3.29637908, 25.86136036],
                [ 1.9550199 ,  8.32190075, 35.98916544],
                [ 2.24482935,  0.38352943, 78.38813322]]])
    """
    
    def __init__(self, sampler, scipy_distribs):
        r"""
        Args:
            sampler (AbstractDiscreteDistribution): A discrete distribution from which to transform samples.
            scipy_distribs (list): instantiated *continuous univariate* distributions from [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions).
        """
        self.domain = np.array([[0,1]])
        if not isinstance(sampler,AbstractDiscreteDistribution):
            raise ParameterError("SciPyWrapper requires sampler be an AbstractDiscreteDistribution.")
        self._parse_sampler(sampler)
        self.scipy_distrib = list(scipy_distribs) if not isinstance(scipy_distribs,scipy.stats._distn_infrastructure.rv_continuous_frozen) else [scipy_distribs]
        for sd in self.scipy_distrib:
            if isinstance(sd,scipy.stats._distn_infrastructure.rv_continuous_frozen): continue
            raise ParameterError('''
                SciPyWrapper requires each value of scipy_distribs to be a 
                1 dimensional scipy.stats continuous distribution, 
                see https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions.''')
        self.sds = self.scipy_distrib if len(self.scipy_distrib)>1 else self.scipy_distrib*sampler.d
        if len(self.sds)!=sampler.d:
            raise DimensionError("length of scipy_distribs must match the dimension of the sampler")
        self.range = np.array([sd.interval(1) for sd in self.sds])
        super(SciPyWrapper,self).__init__()
        assert len(self.sds)==self.d and all(isinstance(sdsi,scipy.stats._distn_infrastructure.rv_continuous_frozen) for sdsi in self.sds)

    def _transform(self, x):
        t = np.empty_like(x) 
        for j in range(self.d):
            t[...,j] = self.sds[j].ppf(x[...,j])
        return t
    
    def _weight(self, x):
        rho = np.empty_like(x)
        for j in range(self.d):
            rho[...,j] = self.sds[j].pdf(x[...,j])
        return np.prod(rho,-1)
    
    def _spawn(self, sampler, dimension):
        return SciPyWrapper(sampler,self.scipy_distrib)
    
