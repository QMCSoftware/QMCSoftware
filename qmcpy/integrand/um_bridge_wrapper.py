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

    >>> import umbridge
    >>> _ = os.system('docker run --name muqbp -dit -p 4243:4243 linusseelinger/benchmark-muq-beam-propagation:latest > /dev/null')
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
    >>> class TestModel(umbridge.Model):
    ...     def __init__(self):
    ...         super().__init__("forward")
    ...     def get_input_sizes(self, config):
    ...         return [1,2,3]
    ...     def get_output_sizes(self, config):
    ...         return [3,2,1]
    ...     def __call__(self, parameters, config):
    ...         out0 = [parameters[2][0],sum(parameters[2][:2]),sum(parameters[2])]
    ...         out1 = [parameters[1][0],sum(parameters[1])]
    ...         out2 = [parameters[0]]
    ...         return [out0,out1,out2]
    ...     def supports_evaluate(self):
    ...         return True
    >>> my_model = TestModel()
    >>> my_distribution = Uniform(
    ...     sampler = DigitalNetB2(dimension=sum(my_model.get_input_sizes(config={})),seed=7),
    ...     lower_bound = -1,
    ...     upper_bound = 1)
    >>> my_integrand = UMBridgeWrapper(my_distribution,my_model)
    >>> my_solution,my_data = CubQMCNetG(my_integrand,abs_tol=5e-2).integrate()
    >>> my_data
    LDTransformData (AccumulateData Object)
        solution        [-2.328e-10 -4.657e-10 -6.985e-10 -2.328e-10 -4.657e-10 -2.328e-10]
        comb_bound_low  [-9.649e-06 -1.765e-04 -2.352e-04 -3.053e-07 -1.997e-05 -1.952e-05]
        comb_bound_high [9.649e-06 1.765e-04 2.352e-04 3.048e-07 1.997e-05 1.952e-05]
        comb_flags      [ True  True  True  True  True  True]
        n_total         2^(10)
        n               [1024. 1024. 1024. 1024. 1024. 1024.]
        time_integrate  ...
    CubQMCNetG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    UMBridgeWrapper (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     -1
        upper_bound     1
    DigitalNetB2 (DiscreteDistribution Object)
        d               6
        dvec            [0 1 2 3 4 5]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()
    >>> my_integrand.to_umbridge_out_sizes(my_solution)
    [[-2.3283064365386963e-10, -4.656612873077393e-10, -6.984919309616089e-10], [-2.3283064365386963e-10, -4.656612873077393e-10], [-2.3283064365386963e-10]]
    >>> my_integrand.to_umbridge_out_sizes(my_data.comb_bound_low)
    [[-9.649316780269146e-06, -0.00017654551993473433, -0.00023524149401055183], [-3.0527962735504843e-07, -1.997367189687793e-05], [-1.9521190552040935e-05]]
    >>> my_integrand.to_umbridge_out_sizes(my_data.comb_bound_high)
    [[9.648851118981838e-06, 0.00017654458861215971, 0.0002352400970266899], [3.048139660677407e-07, 1.9972740574303316e-05], [1.9520724890753627e-05]]

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
        self.d_in_umbridge =  append(0,cumsum(self.model.get_input_sizes(self.config)))
        self.n_d_in_umbridge = len(self.d_in_umbridge)-1
        if self.d_in_umbridge[-1]!=self.true_measure.d:
            raise ParameterError("sampler dimension must equal the sum of UMBridgeWrapper input sizes.")
        self.d_out_umbridge = append(0,cumsum(self.model.get_output_sizes(self.config)))
        self.n_d_out_umbridge = len(self.d_out_umbridge)-1
        super(UMBridgeWrapper,self).__init__(
            dimension_indv = int(self.d_out_umbridge[-1]),
            dimension_comb = int(self.d_out_umbridge[-1]),
            parallel = self.parallel,
            threadpool = True)
    
    def g(self, t, **kwargs):
        n = len(t)
        y = zeros((n,self.d_indv[0]),dtype=float)
        for i in range(n):
            ti_ll = [t[i,self.d_in_umbridge[j]:self.d_in_umbridge[j+1]].tolist() for j in range(self.n_d_in_umbridge)]
            yi_ll = self.model(ti_ll,self.config)
            for j,yi_l in enumerate(yi_ll): y[i,self.d_out_umbridge[j]:self.d_out_umbridge[j+1]] = yi_l if len(yi_l)>1 else yi_l[0]
        return y
    
    def _spawn(self, level, sampler):
        return UMBridgeWrapper(
            true_measure = self.true_measure,
            model = self.model,
            config = self.config, 
            parallel = self.parallel)

    def to_umbridge_out_sizes(self, attr):
        """
        Convert a data attribute to UM-Bridge output sized list of lists. 
        
        Args:
            attr (ndarray): array of length sum(model.get_output_sizes(self.config))
        
        Return:
            list: list of lists with sub-list lengths specified by model.get_output_sizes(self.config)
        """
        return [attr[self.d_out_umbridge[j]:self.d_out_umbridge[j+1]].tolist() for j in range(self.n_d_out_umbridge)]
