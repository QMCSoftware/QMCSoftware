import numpy as np
from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Uniform
from ..util import ParameterError

class Ishigami(Integrand):
    """
    $g(\\boldsymbol{t}) = (1+bt_2^4)\\sin(t_0) + a\\sin^2(t_1), \\qquad T \\sim \\mathcal{U}(-\\pi,\\pi)^3$

    [https://www.sfu.ca/~ssurjano/ishigami.html](https://www.sfu.ca/~ssurjano/ishigami.html)
    
    Examples:
        >>> ishigami = Ishigami(DigitalNetB2(3,seed=7))
        >>> x = ishigami.discrete_distrib.gen_samples(2**10)
        >>> y = ishigami.f(x)
        >>> print("%.4f"%y.mean())
        3.4985
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
            a (float): first paramter
            b (float): second parameter
        """
        self.sampler = sampler
        if self.sampler.d != 3:
            raise ParameterError("Ishigami integrand requires 3 dimensional sampler")
        self.a = a
        self.b = b
        self.true_measure = Uniform(self.sampler, lower_bound=-np.pi, upper_bound=np.pi)
        super(Ishigami,self).__init__(dimension_indv=1,dimension_comb=1,parallel=False)
    
    def g(self, t):
        y = (1+self.b*t[:,2]**4)*np.sin(t[:,0])+self.a*np.sin(t[:,1])**2
        return y

    def _spawn(self, level, sampler):
        return Ishigami(sampler=sampler,a=self.a,b=self.b)
    
    @staticmethod
    def _exact_sensitivity_indices(indices,a,b):
        a,b = np.atleast_1d(a),np.atleast_1d(b)
        assert a.shape==b.shape and a.ndim==1 and b.ndim==1
        mu = a/2
        m2 = 1/2+3/8*a**2+np.pi**4/5*b+np.pi**8/18*b**2
        tau_closed = {
            repr([0]): 1/50*(5+np.pi**4*b)**2,
            repr([1]): a**2/8,
            repr([2]): 0,
            repr([0,1]): a**2/8+1/50*(5+np.pi**4*b)**2,
            repr([0,2]): 1/90*(45+18*np.pi**4*b+5*np.pi**8*b**2),
            repr([1,2]): a**2/8}
        tau_total = {
            repr([1,2]): a**2/8+8*np.pi**8/225*b**2,
            repr([0,2]): 1/90*(45+18*np.pi**4*b+5*np.pi**8*b**2),
            repr([0,1]): 1/2+a**2/8+np.pi**4/5*b+np.pi**8/18*b**2,
            repr([2]): 8*np.pi**8/225*b**2,
            repr([1]): a**2/8,
            repr([0]): 1/90*(45+18*np.pi**4*b+5*np.pi**8*b**2)}
        solution = np.zeros((2,len(indices),len(a)),dtype=float)
        for j,idx in enumerate(indices):
            solution[0,j] = tau_closed[repr(idx)]/(m2-mu**2)
            solution[1,j] = tau_total[repr(idx)]/(m2-mu**2)
        return solution.squeeze()
        
    @staticmethod
    def _exact_fu_functions(x,indices,a,b):
        x = np.atleast_2d(x)
        n = len(x)
        a,b = np.atleast_1d(a),np.atleast_1d(b)
        assert x.ndim==2 and x.shape==(n,3) and a.shape==(1,) and b.shape==(1,)
        x0,x1,x2 = x[:,0],x[:,1],x[:,2]
        fus = {
            repr([]): a/2,
            repr([0]): -1/5*(5+np.pi**4*b)*np.sin(2*np.pi*x0),
            repr([1]): -1/2*a*np.cos(4*np.pi*x1),
            repr([2]): 0,
            repr([0,1]): 0,
            repr([0,2]): -4/5*b*np.pi**4*(1+10*(-1+x2)*x2*(1+2*(-1+x2)*x2))*np.sin(2*np.pi*x0),
            repr([1,2]): -1/2*a*np.cos(4*np.pi*x1)+1/5*(5+b*np.pi**4)*np.sin(2*np.pi*x2),
            repr([0,1,2]): 1/2*a*np.cos(4*np.pi*x1)-1/5*(5+b*np.pi**4)*np.sin(2*np.pi*x1)}
        solution = np.zeros((n,len(indices)),dtype=float)
        for j,idx in enumerate(indices): solution[:,j] = fus[repr(idx)]
        return solution.squeeze()
