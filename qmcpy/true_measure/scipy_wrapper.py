from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
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
        array([[ 2.26535046,  6.36077755, 46.10334984],
               [ 1.37949875,  0.8419074 , 38.66873073],
               [ 2.79562889, -0.65019733, 17.63758514],
               [ 1.91172032,  4.67357136, 55.20163754]])

        >>> true_measure = SciPyWrapper(sampler=DigitalNetB2(2,seed=7),scipy_distribs=scipy.stats.beta(a=5,b=1))
        >>> true_measure(4)
        array([[0.93683292, 0.98238098],
               [0.69611329, 0.84454476],
               [0.99733838, 0.50958933],
               [0.84451252, 0.89011392]])

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
        array([[[ 1.49306553, -0.62826013, 49.7677589 ],
                [ 2.36305807,  4.6683678 , 36.07674167],
                [ 1.96279709,  6.34058526, 21.99536017],
                [ 2.8308265 ,  0.84704612, 51.41443437]],
        <BLANKLINE>
               [[ 1.89753782,  7.30327854, 38.9039967 ],
                [ 2.07271848, -3.84426539, 32.72574799],
                [ 1.46428286,  0.81928225, 21.13131114],
                [ 2.50591431,  4.03840677, 51.08541774]]])
    """

    def __init__(self, sampler, scipy_distribs):
        r"""
        Args:
            sampler (AbstractDiscreteDistribution): A discrete distribution from which to transform samples.
            scipy_distribs (list): instantiated *continuous univariate* distributions from [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions).
        """
        self.domain = np.array([[0, 1]])
        if not isinstance(sampler, AbstractDiscreteDistribution):
            raise ParameterError(
                "SciPyWrapper requires sampler be an AbstractDiscreteDistribution."
            )
        self._parse_sampler(sampler)
        self.scipy_distrib = (
            list(scipy_distribs)
            if not isinstance(
                scipy_distribs, scipy.stats._distn_infrastructure.rv_continuous_frozen
            )
            else [scipy_distribs]
        )
        for sd in self.scipy_distrib:
            if isinstance(sd, scipy.stats._distn_infrastructure.rv_continuous_frozen):
                continue
            raise ParameterError(
                """
                SciPyWrapper requires each value of scipy_distribs to be a 
                1 dimensional scipy.stats continuous distribution, 
                see https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions."""
            )
        self.sds = (
            self.scipy_distrib
            if len(self.scipy_distrib) > 1
            else self.scipy_distrib * sampler.d
        )
        if len(self.sds) != sampler.d:
            raise DimensionError(
                "length of scipy_distribs must match the dimension of the sampler"
            )
        self.range = np.array([sd.interval(1) for sd in self.sds])
        super(SciPyWrapper, self).__init__()
        assert len(self.sds) == self.d and all(
            isinstance(sdsi, scipy.stats._distn_infrastructure.rv_continuous_frozen)
            for sdsi in self.sds
        )

    def _transform(self, x):
        t = np.empty_like(x)
        for j in range(self.d):
            t[..., j] = self.sds[j].ppf(x[..., j])
        return t

    def _weight(self, x):
        rho = np.empty_like(x)
        for j in range(self.d):
            rho[..., j] = self.sds[j].pdf(x[..., j])
        return np.prod(rho, -1)

    def _spawn(self, sampler, dimension):
        return SciPyWrapper(sampler, self.scipy_distrib)
