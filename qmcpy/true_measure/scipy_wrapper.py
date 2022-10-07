from ._true_measure import TrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..discrete_distribution import DigitalNetB2
from numpy import *
import scipy.stats


class SciPyWrapper(TrueMeasure):
    """
    Multivariate True Measure from Independent SciPy 1 Dimensional Marginals

    >>> betas_2d = SciPyWrapper(discrete_distrib=DigitalNetB2(2,seed=7),scipy_distribs=[scipy.stats.beta(a=5,b=1)])
    >>> betas_2d.gen_samples(4)
    array([[0.89136146, 0.70469298],
           [0.80905676, 0.91764986],
           [0.96126183, 0.99081392],
           [0.63619813, 0.86865531]])
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
    array([[ 2.12538017, -0.75733093, 38.28471046],
           [ 1.693306  ,  4.54891231, 80.23287215],
           [ 2.64149095,  9.77761625, 43.6883765 ],
           [ 1.20844522,  2.94566431, 22.68122716]])
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
        self.domain = array([[0,1]])
        if not isinstance(discrete_distrib,DiscreteDistribution):
            raise ParameterError("SciPyWrapper requires discrete_distrib be a DiscreteDistribution.")
        self._parse_sampler(discrete_distrib)
        self.scipy_distrib = list(scipy_distribs)
        for sd in self.scipy_distrib:
            if isinstance(sd,scipy.stats._distn_infrastructure.rv_continuous_frozen): continue
            raise ParameterError('''
                SciPyWrapper requires each value of scipy_distribs to be a 
                1 dimensional scipy.stats continuous distribution, 
                see https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions.''')
        self.sds = self.scipy_distrib if len(self.scipy_distrib)>1 else self.scipy_distrib*discrete_distrib.d
        if len(self.sds)!=discrete_distrib.d:
            raise DimensionError("length of scipy_distribs must match the dimension of the discrete_distrib")
        self.range = array([sd.interval(1) for sd in self.sds])
        super(SciPyWrapper,self).__init__()

    def _transform(self, x):
        return array([self.sds[j].ppf(x[:,j]) for j in range(self.d)],dtype=float).T
    
    def _weight(self, x):
        return array([self.sds[j].pdf(x[:,j]) for j in range(self.d)],dtype=float).T.prod(1)
    
    def _spawn(self, sampler, dimension):
        return SciPyWrapper(sampler,self.scipy_distrib)
    