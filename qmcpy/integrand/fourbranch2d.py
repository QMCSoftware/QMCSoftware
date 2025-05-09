import numpy as np
from .abstract_integrand import AbstractIntegrand
from ..true_measure import Uniform
from ..discrete_distribution import DigitalNetB2


class FourBranch2d(AbstractIntegrand):
    r"""
    Four Branch function in $d=2$. 

    $$g(\boldsymbol{t}) = \min \begin{cases} 3+0.1(t_0-t_1)^2-\frac{t_0-t_1}{\sqrt{2}} \\ 3+0.1(t_0-t_1)^2+\frac{t_0-t_1}{\sqrt{2}} \\ t_0-t_1 + 7/\sqrt{2} \\ t_1-t_0 + 7/\sqrt{2}\end{cases}, \qquad \boldsymbol{T}=(T_0,T_1) \sim \mathcal{U}[-8,8]^2.$$
    
    Examples:
        >>> integrand = FourBranch2d(DigitalNetB2(2,seed=7))
        >>> y = integrand(2**10)
        >>> print("%.4f"%y.mean())
        -2.5022
        >>> integrand.true_measure
        Uniform (AbstractTrueMeasure)
            lower_bound     -8
            upper_bound     2^(3)
            
        With independent replications

        >>> integrand = FourBranch2d(DigitalNetB2(2,seed=7,replications=2**4))
        >>> y = integrand(2**6)
        >>> y.shape
        (16, 64)
        >>> muhats = y.mean(-1) 
        >>> muhats.shape 
        (16,)
        >>> print("%.4f"%muhats.mean())
        -2.5147
    """

    def __init__(self, sampler):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
        """
        self.sampler = sampler
        assert self.sampler.d==2
        self.true_measure = Uniform(self.sampler,lower_bound=-8,upper_bound=8)
        super(FourBranch2d,self).__init__(dimension_indv=(),dimension_comb=(),parallel=False)
    
    def g(self, t):
        t0,t1 = t[...,0],t[...,1]
        return np.minimum.reduce([
            3+.1*(t0-t1)**2-(t0+t1)/np.sqrt(2),
            3+.1*(t0-t1)**2+(t0+t1)/np.sqrt(2),
            t0-t1+7/np.sqrt(2),
            t1-t0+7/np.sqrt(2)])
    
    def _spawn(self, level, sampler):
        return FourBranch2d(sampler=sampler) 
