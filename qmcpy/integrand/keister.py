from .abstract_integrand import AbstractIntegrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian
import numpy as np
from scipy.special import gamma


class Keister(AbstractIntegrand):
    """
    $f(\\boldsymbol{t}) = \\pi^{d/2} \\cos(\\| \\boldsymbol{t} \\|)$.

    The standard example integrates the Keister integrand with respect to an 
    IID Gaussian distribution with variance 1./2.

    >>> k = Keister(DigitalNetB2(2,seed=7))
    >>> x = k.discrete_distrib.gen_samples(2**10)
    >>> y = k.f(x)
    >>> print("%.4f"%y.mean())
    1.8082
    >>> k.true_measure
    Gaussian (AbstractTrueMeasure Object)
        mean            0
        covariance      2^(-1)
        decomp_type     PCA
    >>> k = Keister(Gaussian(DigitalNetB2(2,seed=7),mean=0,covariance=2))
    >>> x = k.discrete_distrib.gen_samples(2**12)
    >>> y = k.f(x)
    >>> print("%.4f"%y.mean())
    1.8081
    >>> yp = k.f(x,periodization_transform='c2sin')
    >>> print("%.4f"%yp.mean())
    1.8080

    References:

        [1] B. D. Keister, Multidimensional Quadrature Algorithms, 
        `Computers in Physics`, *10*, pp. 119-122, 1996.
    """

    def __init__(self, sampler):
        """
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
        """
        self.sampler = sampler
        self.true_measure = Gaussian(self.sampler,mean=0,covariance=1/2)
        super(Keister,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
    
    def g(self, t):
        d = t.shape[1]
        norm = np.sqrt((t**2).sum(1))
        k = np.pi**(d/2)*np.cos(norm)
        return k
    
    def _spawn(self, level, sampler):
        return Keister(sampler=sampler)

    def exact_integ(self, d):
        """
        computes the true value of the Keister integral in dimension d.
        Accuracy might degrade as d increases due to round-off error.
        :param d:
        :return: true_integral
        """
        cosinteg = np.zeros(shape=(d))
        cosinteg[0] = np.sqrt(np.pi) / (2 * np.exp(1 / 4))
        sininteg = np.zeros(shape=(d))
        sininteg[0] = 4.244363835020225e-01
        cosinteg[1] = (1 - sininteg[0]) / 2
        sininteg[1] = cosinteg[0] / 2
        for j in range(2, d):
            cosinteg[j] = ((j-1)*cosinteg[j-2]-sininteg[j-1])/2
            sininteg[j] = ((j-1)*sininteg[j-2]+cosinteg[j-1])/2
        I = (2*(np.pi**(d/2))/gamma(d/2))*cosinteg[d-1]
        return I
