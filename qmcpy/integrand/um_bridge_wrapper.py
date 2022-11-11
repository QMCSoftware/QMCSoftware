from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..stopping_criterion import CubQMCNetG
from ..true_measure import Uniform
from ..util import ParameterError
from numpy import *
import os

class UMBridgeWrapper(Integrand):
    """
    UM-Bridge Model Wrapper

    >>> _ = os.system('docker run --name muqbp -dit -p 4243:4243 linusseelinger/benchmark-muq-beam-propagation:latest > /dev/null')
    >>> import umbridge
    >>> dnb2 = DigitalNetB2(dimension=3,seed=7)
    >>> distribution = Uniform(dnb2,lower_bound=1,upper_bound=1.05)
    >>> model = umbridge.HTTPModel('http://localhost:4243','forward')
    >>> umbridge_config = {"d": dnb2.d}
    >>> um_bridge_integrand = UMBridgeWrapper(distribution,model,umbridge_config,parallel=False)
    >>> solution,data = CubQMCNetG(um_bridge_integrand,abs_tol=5e-2).integrate()
    >>> print(data)
    LDTransformData (AccumulateData Object)
        solution        [  0.      3.855  14.69  ... 898.921 935.383 971.884]
        comb_bound_low  [  0.      3.854  14.688 ... 898.901 935.363 971.863]
        comb_bound_high [  0.      3.855  14.691 ... 898.941 935.404 971.906]
        comb_flags      [ True  True  True ...  True  True  True]
        n_total         2^(11)
        n               [1024. 1024. 1024. ... 2048. 2048. 2048.]
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    UMBridgeWrapper (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     1
        upper_bound     1.050
    DigitalNetB2 (DiscreteDistribution Object)
        d               3
        dvec            [0 1 2]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()
    >>> _ = os.system('docker rm -f muqbp > /dev/null')

    References:

        [1] UM-Bridge documentation. https://um-bridge-benchmarks.readthedocs.io/en/docs/index.html
    """

    def __init__(self, true_measure, model, config={}, parallel=False):
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
        if not self.model.supports_evaluate(): raise ParameterError("UMBridgeWrapper requires model supports evaluation.")
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