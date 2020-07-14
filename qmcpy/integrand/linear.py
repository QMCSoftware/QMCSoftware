from ._integrand import Integrand
from ..discrete_distribution import Sobol
from ..true_measure import Gaussian


class Linear(Integrand):
    """
    $f(\\boldsymbol{x}) = \\sum_{i=1}^d x_i$
    
    >>> dd = Sobol(100,seed=7)
    >>> m = Gaussian(dd,covariance=1./3)
    >>> l = Linear(m)
    >>> x = dd.gen_samples(2**10)
    >>> y = l.f(x)
    >>> y.mean()
    -0.0008447105940667977
    """

    def __init__(self, measure):
        """
        Args:
            measure (TrueMeasure): a TrueMeasure instance
        """
        self.measure = measure
        super(Linear,self).__init__()

    def g(self, x):
        """ See abstract method. """
        y = x.sum(1)  # Linear sum
        return y
