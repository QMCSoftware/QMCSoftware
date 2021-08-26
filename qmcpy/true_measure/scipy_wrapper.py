from ._true_measure import TrueMeasure
from ..util import TransformError, DimensionError
from ..discrete_distribution import DigitalNetB2
from numpy import *
import scipy.stats


class SciPyWrapper(TrueMeasure):
    """
    >>> triangular = SciPyWrapper(
    ...     sampler = DigitalNetB2(2,seed=7),
    ...     scipy_distrib = scipy.stats.triang,
    ...     c = [0.1,.2],
    ...     loc = [1,2],
    ...     scale = [3,4])
    >>> triangular.gen_samples(4)
    array([[2.11792393, 2.74571838],
           [1.6995412 , 3.88553573],
           [2.79502629, 5.24025887],
           [1.30634136, 3.45650562]])
    >>> triangular
    SciPyWrapper (TrueMeasure Object)
        scipy_distrib   triang
        c               [0.1 0.2]
        loc             [1 2]
        scale           [3 4]
    """
    
    def __init__(self, sampler, scipy_distrib, **scipy_distrib_kwargs):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            scipy_stat_distrib (float): a CONTINUOUS UNIVARIATE scipy.stats distribution e.g. scipy.stats.norm, 
                see https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions
            **scipy_distrib_kwargs (float): keyword arguments for scipy_stat_distrib.{pdf,ppf or pmf}. 
                Note that you may pass in vectors of keyword arguments and they will be distributed 
                appropriotly across the dimensions. 
                Also note that positional aguments to scipy_stat_distrib.{pdf,ppf or pmf} must still 
                be supplied as keyword arguments to QMCPy's SciPyWrapper, see e.g. the above doctest and 
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html#scipy.stats.triang
        """
        self.scipy_distrib_kwargs = scipy_distrib_kwargs
        self.kwarg_keys = list(self.scipy_distrib_kwargs.keys())
        self.parameters = ['scipy_distrib']+self.kwarg_keys
        self.sd = scipy_distrib
        self.scipy_distrib = type(self.sd).__name__.replace('_gen','')
        self.domain = array([[0,1]])
        self._parse_sampler(sampler)
        for key in self.kwarg_keys:
            val = self.scipy_distrib_kwargs[key]
            if isscalar(val):
                val = tile(val,self.d)
            val = array(val)
            if val.shape != (self.d,):
                raise DimensionError('''
                    %s input must be an ndarray with length = dimension,
                    or a scalar that will be copied to length dimension.''')
            self.scipy_distrib_kwargs[key] = val
            setattr(self,key,val)
        self.range = zeros((self.d,2),dtype=float)
        self.kwargs_dims = [{key:self.scipy_distrib_kwargs[key][j] for key in self.kwarg_keys} for j in range(self.d)]
        self.range = array([self.sd.interval(alpha=1,**self.kwargs_dims[j]) for j in range(self.d)],dtype=float)
        super(SciPyWrapper,self).__init__()

    def _transform(self, x):
        return array([self.sd.ppf(x[:,j],**self.kwargs_dims[j]) for j in range(self.d)],dtype=float).T

    def _jacobian(self, x):
        return array([
            1/self.sd.pdf(self.sd.ppf(x[:,j],**self.kwargs_dims[j]),**self.kwargs_dims[j])
            for j in range(self.d)],dtype=float).T.prod(1)
    
    def _weight(self, x):
        return array([self.sd.pdf(x[:,j],**self.kwargs_dims[j]) for j in range(self.d)],dtype=float).T.prod(1)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = SciPyWrapper(sampler,self.sd,**self.scipy_distrib_kwargs)
        else:
            new_kwargs = {}
            for key in self.kwarg_keys:
                val = self.scipy_distrib_kwargs[key]
                v0 = val[0]
                if not all(val==v0):
                    raise DimensionError('''
                        In order to spawn a SciPyWrapper measure
                        each keyword argument must be a tiled (repeated) scalar, 
                        otherwise, QMCPy doesn't know what values to use for new dimensions.''')
                new_kwargs[key] = tile(v0,dimension)
            spawn = SciPyWrapper(sampler,self.sd,**new_kwargs)
        return spawn
    