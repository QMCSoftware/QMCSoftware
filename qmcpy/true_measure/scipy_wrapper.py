from ._true_measure import TrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from ..discrete_distribution import DigitalNetB2
import numpy as np
import scipy.stats


class SciPyWrapper(TrueMeasure):
    """
    Multivariate True Measure from Independent SciPy 1 Dimensional Marginals

    >>> unif_gauss_gamma = SciPyWrapper(
    ...     discrete_distrib = DigitalNetB2(3,seed=7),
    ...     scipy_distribs = [
    ...         scipy.stats.uniform(loc=1,scale=2),
    ...         scipy.stats.norm(loc=3,scale=4),
    ...         scipy.stats.gamma(a=5,loc=6,scale=7)])
    >>> unif_gauss_gamma.range
    array([[  1.,   3.],
           [-inf,  inf],
           [  6.,  inf]])
    >>> unif_gauss_gamma.gen_samples(4)
    array([[ 2.53470309,  0.35273007, 56.37744157],
           [ 1.35215166,  7.42509911, 33.50121705],
           [ 2.04111416,  3.83592672, 28.72235698],
           [ 1.8447669 , -4.11180523, 48.44339757]])
    >>> betas_2d = SciPyWrapper(discrete_distrib=DigitalNetB2(2,seed=7),scipy_distribs=scipy.stats.beta(a=5,b=1))
    >>> betas_2d.gen_samples(4)
    array([[0.59010522, 0.60012506],
           [0.9597241 , 0.9427888 ],
           [0.79325286, 0.98608598],
           [0.89417617, 0.76694864]])
    """
    
    def __init__(self, discrete_distrib, scipy_distribs):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            scipy_distribs (list): instantiated CONTINUOUS UNIVARIATE scipy.stats distributions 
                https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
        """
        self.domain = np.array([[0,1]])
        if not isinstance(discrete_distrib,DiscreteDistribution):
            raise ParameterError("SciPyWrapper requires discrete_distrib be a DiscreteDistribution.")
        self._parse_sampler(discrete_distrib)
        self.scipy_distrib = list(scipy_distribs) if not isinstance(scipy_distribs,scipy.stats._distn_infrastructure.rv_continuous_frozen) else [scipy_distribs]
        for sd in self.scipy_distrib:
            if isinstance(sd,scipy.stats._distn_infrastructure.rv_continuous_frozen): continue
            raise ParameterError('''
                SciPyWrapper requires each value of scipy_distribs to be a 
                1 dimensional scipy.stats continuous distribution, 
                see https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions.''')
        self.sds = self.scipy_distrib if len(self.scipy_distrib)>1 else self.scipy_distrib*discrete_distrib.d
        if len(self.sds)!=discrete_distrib.d:
            raise DimensionError("length of scipy_distribs must match the dimension of the discrete_distrib")
        self.range = np.array([sd.interval(1) for sd in self.sds])
        super(SciPyWrapper,self).__init__()

    def _transform(self, x):
        return np.array([self.sds[j].ppf(x[:,j]) for j in range(self.d)],dtype=float).T
    
    def _weight(self, x):
        return np.array([self.sds[j].pdf(x[:,j]) for j in range(self.d)],dtype=float).T.prod(1)
    
    def _spawn(self, sampler, dimension):
        return SciPyWrapper(sampler,self.scipy_distrib)
    
