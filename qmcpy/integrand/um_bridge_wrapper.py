from ._integrand import Integrand
from ..util import ParameterError
from numpy import *

class UMBridgeWrapper(Integrand):
    """
    UM-Bridge Model Wrapper
    References:

        [1] UM-Bridge documentation. https://um-bridge-benchmarks.readthedocs.io/en/docs/index.html
    """

    def __init__(self, true_measure, model, config, parallel=False):
        """
        See https://um-bridge-benchmarks.readthedocs.io/en/docs/umbridge/clients.html
        
        Args:
            true_measure (TrueMeasure): a TrueMeasure instance. 
            model (umbridge.HTTPModel): a UM-Bridge model 
            config (dict): config keyword argument to umbridge.HTTPModel(url,name).__call__
            parallel (int): If parallel is False, 0, or 1: function evaluation is done in serial fashion.
                Otherwise, parallel specifies the number of processes used by 
                multiprocessing.Pool or multiprocessing.pool.ThreadPool.
                Passing parallel=True sets processes = os.cpu_count().
        """
        import umbridge
        self.parameters = []
        self.true_measure = true_measure
        self.sampler = self.true_measure 
        self.model = model
        self.config = config
        self.parallel = parallel
        d_umbridge = self.model.get_input_sizes(self.config)[0]
        if d_umbridge!=self.true_measure.d:
            raise ParameterError("input dimension to umbridge model must match true_measure dimension.")
        dimension_indv = self.model.get_output_sizes(self.config)[0]
        super(UMBridgeWrapper,self).__init__(
            dimension_indv = dimension_indv,
            dimension_comb = dimension_indv,
            parallel = self.parallel,
            threadpool = True)
    
    def g(self, t, **kwargs):
        return array([self.model([t[i].tolist()],self.config)[0] for i in range(len(t))],dtype=float)
    
    def _spawn(self, level, sampler):
        return UMBridgeWrapper(
            true_measure = self.true_measure,
            model = self.model,
            config = self.config, 
            parallel = self.parallel)