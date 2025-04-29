from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian, Uniform


class CustomFun(Integrand):
    """
    Integrand wrapper for a user's function 
    
    >>> cf = CustomFun(
    ...     true_measure = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2]),
    ...     g = lambda x: x[:,0]**2*x[:,1],
    ...     dimension_indv = 1)
    >>> x = cf.discrete_distrib.gen_samples(2**10)
    >>> y = cf.f(x)
    >>> y.shape
    (1024, 1)
    >>> y.mean()
    np.float64(3.9901816652680573)
    >>> cf = CustomFun(
    ...     true_measure = Uniform(DigitalNetB2(3,seed=7),lower_bound=[2,3,4],upper_bound=[4,5,6]),
    ...     g = lambda x,compute_flags=None: x,
    ...     dimension_indv = 3)
    >>> x = cf.discrete_distrib.gen_samples(2**10)
    >>> y = cf.f(x)
    >>> y.shape
    (1024, 3)
    >>> y.mean(0)
    array([3., 4., 5.])
    """

    def __init__(self, true_measure, g, dimension_indv=1, parallel=False):
        """
        Args:
            true_measure (TrueMeasure): a TrueMeasure instance. 
            g (function): a function handle. 
            dimension_indv (tuple): individual solution dimensions.
            parallel (int): If parallel is False, 0, or 1: function evaluation is done in serial fashion. 
                Otherwise, parallel specifies the number of CPUs used by multiprocessing.Pool. 
                Passing parallel=True sets the number of CPUs equal to os.cpu_count(). 
                Do NOT set g to a lambda function when doing parallel computation
        """
        self.parameters = []
        self.true_measure = true_measure
        self.sampler = self.true_measure
        self.__g = g            
        super(CustomFun,self).__init__(dimension_indv=dimension_indv,dimension_comb=dimension_indv,parallel=parallel)
    
    def g(self, t, *args, **kwargs):
        return self.__g(t,*args,**kwargs)
    
    def _spawn(self, level, sampler):
        return CustomFun(
            true_measure = sampler,
            g = self.__g,
            dimension_indv = self.d_indv,
            parallel = self.parallel)
