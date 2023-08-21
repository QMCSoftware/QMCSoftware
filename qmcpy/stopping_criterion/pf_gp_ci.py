from ._stopping_criterion import StoppingCriterion
from ..discrete_distribution import DigitalNetB2
from ..integrand.ishigami import Ishigami
from ..true_measure._true_measure import TrueMeasure
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..accumulate_data.pf_gp_ci_data import PFGPCIData, _error_udens
from ..util import MaxSamplesWarning
import warnings
import time
from numpy import *
from scipy.stats import norm
try:
    import gpytorch 
    import torch
except: pass 

class Suggester(object): pass

class PFSampleErrorDensityAR(Suggester):
    def __init__(self, verbose=False):
        self.verbose = verbose
        super(PFSampleErrorDensityAR,self).__init__()
    def suggest(self, n, d, gp, rng, efficiency, pct=.5):
        if self.verbose: print('\tAR sampling with efficiency %.1e, expect %d draws: '%(efficiency,int(n/efficiency)),end='',flush=True)
        x = zeros((0,d),dtype=float)
        self.n_ar_tries_batch = 0
        rem = n
        draws_per_rem = int(ceil(log(pct)/log(1-efficiency)))
        while rem>0:
            newtries = rem*draws_per_rem
            self.n_ar_tries_batch += newtries
            if self.verbose: print('%d, '%self.n_ar_tries_batch,end='',flush=True)
            z = rng.random((newtries,d))
            u = rng.random(newtries)
            udens_z = _error_udens(gp,z)
            x = vstack([x,z[u<=udens_z]])
            rem = max(n-len(x),0)
        if self.verbose: print()
        return x[:n]

class SuggesterSimple(Suggester):
    def __init__(self, sampler):
        self.sampler = sampler
        if isinstance(self.sampler,TrueMeasure): assert (self.sampler.range==[0,1]).all()
        self.n_min = 0
        super(SuggesterSimple,self).__init__()
    def suggest(self, n, d, gp, rng, **kwargs):
        n_max = self.n_min+n
        assert d == self.sampler.d
        try: x = self.sampler.gen_samples(n_min=self.n_min,n_max=n_max)
        except: x = self.sampler.gen_samples(n)
        self.n_min = n_max
        return x

class PFGPCI(StoppingCriterion):
    """
    Probability of failure estimation using adaptive Gaussian Processes (GP) construction and resulting credible intervals. 
    
    >>> pfgpci = PFGPCI(
    ...     integrand = Ishigami(DigitalNetB2(3,seed=17)),
    ...     failure_threshold = 0,
    ...     failure_above_threshold = False, 
    ...     abs_tol = 2.5e-2,
    ...     alpha = 1e-1,
    ...     n_init = 64,
    ...     init_samples = None,
    ...     batch_sampler = PFSampleErrorDensityAR(verbose=False),
    ...     n_batch = 16,
    ...     n_max = 128,
    ...     n_approx = 2**18,
    ...     gpytorch_prior_mean = gpytorch.means.ZeroMean(),
    ...     gpytorch_prior_cov = gpytorch.kernels.ScaleKernel(
    ...         gpytorch.kernels.MaternKernel(nu=2.5,lengthscale_constraint = gpytorch.constraints.Interval(.5,1)),
    ...         outputscale_constraint = gpytorch.constraints.Interval(1e-8,.5)),
    ...     gpytorch_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.Interval(1e-12,1e-8)),
    ...     gpytorch_marginal_log_likelihood_func = lambda likelihood,gpyt_model: gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,gpyt_model),
    ...     torch_optimizer_func = lambda gpyt_model: torch.optim.Adam(gpyt_model.parameters(),lr=0.1),
    ...     gpytorch_train_iter = 100,
    ...     gpytorch_use_gpu = False,
    ...     verbose = False,
    ...     n_ref_approx = 2**22,
    ...     seed_ref_approx = 11)
    >>> solution,data = pfgpci.integrate(seed=7,refit=True)
    >>> data
    PFGPCIData (AccumulateData Object)
        solution        0.161
        error_bound     0.025
        bound_low       0.136
        bound_high      0.186
        n_total         112
        time_integrate  ...
    PFGPCI (StoppingCriterion Object)
    Ishigami (Integrand Object)
    Uniform (TrueMeasure Object)
        lower_bound     -3.142
        upper_bound     3.142
    DigitalNetB2 (DiscreteDistribution Object)
        d               3
        dvec            [0 1 2]
        randomize       LMS_DS
        graycode        0
        entropy         17
        spawn_key       ()
    >>> df = data.get_results_dict()
    """
    def __init__(self, 
        integrand,
        failure_threshold,
        failure_above_threshold,
        abs_tol = 5e-3,
        alpha = 1e-2,
        n_init = 64,
        init_samples = None,
        batch_sampler = PFSampleErrorDensityAR(),
        n_batch = 4,
        n_max = 1000,
        n_approx = 2**20,
        gpytorch_prior_mean = gpytorch.means.ZeroMean(),
        gpytorch_prior_cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 2.5)),
        gpytorch_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = gpytorch.constraints.Interval(1e-12,1e-8)),
        gpytorch_marginal_log_likelihood_func = lambda likelihood,gpyt_model: gpytorch.mlls.ExactMarginalLogLikelihood(likelihood,gpyt_model),
        torch_optimizer_func = lambda gpyt_model: torch.optim.Adam(gpyt_model.parameters(),lr=0.1),
        gpytorch_train_iter = 100,
        gpytorch_use_gpu = False,
        verbose = False,
        n_ref_approx = 2**22,
        seed_ref_approx = None):
        '''
        Args:
            integrand (Integrand): The simulation whose probability of failure is estimated
            failure_threshold (float): Thresholds for failure. 
            failure_above_threshold (bool): Set to True if failure occurs when the simulation exceeds failure_threshold and False otherwise 
            abs_tol (float): The desired maximum distance from the estimate to either end of the confidence interval
            alpha (float): The credible interval is constructed to hold with probability at least 1 - alpha
            n_init (float): Initial number of samples from integrand.discrete_distrib from which to build the first surrogate GP
            init_samples (float): If the simulation has already been run, pass in (x,y) where x are past samples from the discrete distribution and y are corresponding simulation evaluations. 
            batch_sampler (Suggester or DiscreteDistsribution): A suggestion scheme for future samples. 
            n_batch (int): The number of samples per batch to draw from batch_sampler. 
            n_max (int): Budget of simulations.
            n_approx (int): Number of points from integrand.discrete_distrib used to approximate estimate and credible interval bounds
            gpytorch_prior_mean (gpytorch.means): prior mean function of the GP
            gpytorch_prior_cov (gpytorch.kernels): Prior covariance kernel of the GP
            gpytorch_likelihood (gpytorch.likelihoods): GP likelihood, require one of gpytorch.likelihoods.{GaussianLikelihood, GaussianLikelihoodWithMissingObs, FixedNoiseGaussianLikelihood}
            gpytorch_marginal_log_likelihood_func (callable): Function taking in the likelihood and gpytorch model and returning a marginal log likelihood from gpytorch.mlls
            torch_optimizer_func (callable): Function taking in the gpytorch model and returning an optimizer from torch.optim
            gpytorch_train_iter (int): Triaining iterations for the GP in gpytorch
            gpytorch_use_gpu (bool): If True, have gpytorch use a GPU for fitting and trining the GP
            verbose (int): If verbose > 0, print information throught the call to integrate()
            n_ref_approx (int): If n_ref_approx > 0, use n_ref_approx points to get a reference QMC approximation of the true solution. 
                Caution: If n_ref_approx > 0, it should be a large int e.g. 2**22, in which case it is only helpful for cheap to evaluate simulations 
            seed_ref_approx (int): Seed for the reference aproximation. Only applies when n_ref_approx>0
        '''
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.d = self.integrand.d
        self.sampler = self.d
        self.failure_threshold = failure_threshold
        self.failure_above_threshold = failure_above_threshold
        self.abs_tol = abs_tol
        self.alpha = alpha
        self.n_init = n_init
        self.init_samples = init_samples is not None
        if self.init_samples: 
            self.x_init,self.y_init = init_samples
            assert self.x_init.ndim==2 and self.y_init.ndim==1
            assert self.x_init.shape[1]==self.d and len(self.y_init)==len(self.x_init)
            assert self.n_init==len(self.x_init)
            self.ytf_init = self._affine_tf(self.y_init)
        self.batch_sampler = batch_sampler
        self.n_batch = n_batch
        self.n_max = n_max
        assert self.n_max >= self.n_init
        self.n_approx = n_approx
        assert (self.n_approx+self.n_init) <= 2**32
        self.gpytorch_prior_mean = gpytorch_prior_mean
        self.gpytorch_prior_cov = gpytorch_prior_cov
        self.gpytorch_likelihood = gpytorch_likelihood
        self.gpytorch_marginal_log_likelihood_func = gpytorch_marginal_log_likelihood_func
        self.torch_optimizer_func = torch_optimizer_func
        self.gpytorch_train_iter = gpytorch_train_iter
        self.gpytorch_use_gpu = gpytorch_use_gpu
        self.verbose = verbose
        self.approx_true_solution = n_ref_approx>0
        if self.approx_true_solution: 
            x = DigitalNetB2(self.d,graycode=True,seed=seed_ref_approx).gen_samples(n_ref_approx)
            y = self.integrand.f(x).squeeze()
            ytf = self._affine_tf(y)
            self.ref_approx = (ytf>=0).mean(0)
            if self.verbose: print('reference approximation with d=%d: %s'%(self.d,self.ref_approx))
        super(PFGPCI,self).__init__(allowed_levels=["single"], allowed_distribs=[DiscreteDistribution], allow_vectorized_integrals=False)
    def _affine_tf(self, y):
        return y-self.failure_threshold if self.failure_above_threshold else self.failure_threshold-y
    #@profile
    def integrate(self, seed=None, refit=False):
        t0 = time.time()
        dnb2 = DigitalNetB2(self.d,randomize='DS',graycode=True,seed=seed)
        data = PFGPCIData(self,self.integrand,self.true_measure,self.discrete_distrib,dnb2,self.n_approx,self.alpha,refit, 
                self.gpytorch_prior_mean,
                self.gpytorch_prior_cov,
                self.gpytorch_likelihood,
                self.gpytorch_marginal_log_likelihood_func,
                self.torch_optimizer_func,
                self.gpytorch_train_iter,
                self.gpytorch_use_gpu,
                self.verbose, self.approx_true_solution)
        batch_count = 0
        while True:
            if self.verbose: print('batch %d'%batch_count)
            if batch_count==0:
                if self.init_samples: 
                    xdraw,ydraw = self.x_init,self.y_init
                else:
                    xdraw = dnb2.spawn()[0].gen_samples(self.n_init)
                    ydraw = atleast_1d(self.integrand.f(xdraw).squeeze())
            else:
                n_new = min(self.n_batch,self.n_max-sum(self.n_batch))
                xdraw = self.batch_sampler.suggest(n_new,self.d,data.gpyt_model,dnb2.rng,efficiency=2*data.emr[-1])
                ydraw = atleast_1d(self.integrand.f(xdraw).squeeze())
            ydrawtf = self._affine_tf(ydraw)
            data.update_data(batch_count, xdraw, ydrawtf)
            batch_count += 1
            if data.error_bounds[-1] <= self.abs_tol: break
            if sum(data.n_batch)==self.n_max:
                warnings.warn('n_max reached. ',MaxSamplesWarning)
                break        
        data.solution = data.solutions[-1]
        data.error_bound = data.error_bounds[-1]
        data.bound_low = data.ci_low[-1]
        data.bound_high = data.ci_high[-1]
        data.n_total = sum(data.n_batch)
        data.time_integrate = time.time()-t0
        data.n_sum = cumsum(data.n_batch)
        data.n_batch = array(data.n_batch)
        data.error_bounds = array(data.error_bounds)
        data.ci_low = array(data.ci_low)
        data.ci_high = array(data.ci_high)
        data.solutions = array(data.solutions)
        if self.approx_true_solution:
            data.solutions_ref = tile(self.ref_approx,len(data.n_batch))
            data.error_ref = abs(data.solutions-data.solutions_ref)
            data.in_ci = (data.ci_low<=data.solutions_ref)*(data.solutions_ref<=data.ci_high)
        return data.solution,data
