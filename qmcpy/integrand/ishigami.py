from numpy import *
from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
from ..util import ParameterError

class Ishigami(Integrand):
    """
    $g(\\boldsymbol{t}) = (1+bt_2^4)\\sin(t_0) + a\\sin^2(t_1), \\qquad T \\sim \\mathcal{U}(-\\pi,\\pi)^3$

    https://www.sfu.ca/~ssurjano/ishigami.html
    
    
    >>> ishigami = Ishigami(DigitalNetB2(3,seed=7))
    >>> x = ishigami.discrete_distrib.gen_samples(2**10)
    >>> y = ishigami.f(x)
    >>> y.mean()
    3.499...
    >>> ishigami.true_measure
    Uniform (TrueMeasure Object)
        lower_bound     -3.142
        upper_bound     3.142
        
    References
        [1] Ishigami, T., & Homma, T. (1990, December). 
        An importance quantification technique in uncertainty analysis for computer models. 
        In Uncertainty Modeling and Analysis, 1990. Proceedings., First International Symposium on (pp. 398-403). IEEE.

    """
    def __init__(self,sampler, a=7, b=.1):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            a,b (float): fixed parameters in above equation
        """
        self.sampler = sampler
        if self.sampler.d != 3:
            raise ParameterError("Ishigami integrand requires 3 dimensional sampler")
        self.a = a
        self.b = b
        self.true_measure = Uniform(self.sampler, lower_bound=-pi, upper_bound=pi)
        super(Ishigami,self).__init__(dprime=1,parallel=False)
    
    def g(self, t):
        y = (1+self.b*t[:,2]**4)*sin(t[:,0])+self.a*sin(t[:,1])**2
        return y

    def _spawn(self, level, sampler):
        return Ishigami(sampler=sampler,a=self.a,b=self.b)