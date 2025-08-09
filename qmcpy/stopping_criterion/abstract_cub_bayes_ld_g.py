from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..util.data import Data

from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
import numpy as np
from time import time
import warnings
from scipy.optimize import fminbound as fminbnd
from scipy.optimize import fmin, fmin_bfgs
from scipy.stats import norm as gaussnorm
from scipy.stats import t as tnorm


class AbstractCubBayesLDG(AbstractStoppingCriterion):

    def __init__(self, integrand, ft, omega, ptransform, allowed_distribs, kernel,
                 abs_tol, rel_tol, n_init, n_limit, alpha, error_fun, errbd_type):
        self.parameters = ['abs_tol', 'rel_tol', 'n_init', 'n_limit', 'order']
        # Input Checks
        if np.log2(n_init)%1!=0:
            warnings.warn('n_init must be a power of two. Using n_init = 2**5',ParameterWarning)
            n_init = 2**8
        if np.log2(n_limit)%1!=0:
            warnings.warn('n_init must be a power of two. Using n_limit = 2**30',ParameterWarning)
            n_limit = 2**22
        # Set Attributes
        self.n_init = int(n_init)
        self.n_limit = int(n_limit)
        assert isinstance(error_fun,str) or callable(error_fun)
        if isinstance(error_fun,str):
            if error_fun.upper()=="EITHER":
                error_fun = lambda sv,abs_tol,rel_tol: np.maximum(abs_tol,abs(sv)*rel_tol)
            elif error_fun.upper()=="BOTH":
                error_fun = lambda sv,abs_tol,rel_tol: np.minimum(abs_tol,abs(sv)*rel_tol)
            else:
                raise ParameterError("str error_fun must be 'EITHER' or 'BOTH'")
        self.error_fun = error_fun
        self.alpha = alpha
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        super(AbstractCubBayesLDG,self).__init__(allowed_distribs=allowed_distribs,allow_vectorized_integrals=True)
        assert self.integrand.discrete_distrib.no_replications==True, "Require the discrete distribution has replications=None"
        assert self.integrand.discrete_distrib.randomize!="FALSE", "Require discrete distribution is randomized"
        self.alphas_indv,identity_dependency = self._compute_indv_alphas(np.full(self.integrand.d_comb,self.alpha))
        self.set_tolerance(abs_tol,rel_tol)
        self.stop_at_tol = True  # automatic mode: stop after meeting the error tolerance
        self.arb_mean = True  # by default use zero mean algorithm
        self.avoid_cancel_error = True  # avoid cancellation error in stopping criterion
        self.debug_enable = False  # enable debug prints
        self.use_gradient = False  # If true uses gradient descent in parameter search
        self.one_theta = True  # If true use common shape parameter for all dimensions, else allow shape parameter vary across dimensions
        self.errbd_type = errbd_type.upper()  
        assert self.errbd_type in ['MLE',"GCV","FULL"]
        self.kernel = kernel
        self.debugEnable = True
        self.ft = ft
        self.omega = omega
        self.ptransform = ptransform  # periodization transform
        if self.errbd_type == 'FULL':
            self.uncert = -tnorm.ppf(alpha/2,self.n_init-1)
        else:
            self.uncert = -gaussnorm.ppf(alpha / 2)

    def _stopping_criterion(self, xpts, ftilde, m):
        ftilde = ftilde.squeeze()
        n = 2 ** m
        lna_range = [-5, 0]  # reduced from [-5, 5], to avoid kernel values getting too big causing error

        # search for optimal shape parameter
        if self.one_theta == True:
            lna_MLE = fminbnd(lambda lna: self.objective_function(np.exp(lna), xpts, ftilde)[0],
                              x1=lna_range[0], x2=lna_range[1], xtol=1e-2, disp=0)

            aMLE = np.exp(lna_MLE)
            _, vec_lambda, vec_lambda_ring, RKHS_norm = self.objective_function(aMLE, xpts, ftilde)
        else:
            if self.use_gradient == True:
                warnings.warn('Not implemented !')
                lna_MLE = 0
            else:
                # Nelder-Mead Simplex algorithm
                theta0 = np.ones((xpts.shape[1], 1)) * (0.05)
                theta0 = np.ones((1, xpts.shape[1])) * (0.05)
                lna_MLE = fmin(lambda lna: self.objective_function(np.exp(lna), xpts, ftilde)[0],
                               theta0, xtol=1e-2, disp=False)
            aMLE = np.exp(lna_MLE)
            # print(n, aMLE)
            _, vec_lambda, vec_lambda_ring, RKHS_norm = self.objective_function(aMLE, xpts, ftilde)

        # Check error criterion
        # compute DSC
        if self.errbd_type == 'FULL':
            # full Bayes
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / n)
            else:
                DSC = abs((vec_lambda[0] / n) - 1)

            # 1-alpha two sided confidence interval
            err_bd = self.uncert * np.sqrt(DSC * RKHS_norm / (n - 1))
        elif self.errbd_type == 'GCV':
            # GCV based stopping criterion
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / (n + vec_lambda_ring[0]))
            else:
                DSC = abs(1 - (n / vec_lambda[0]))

            temp = vec_lambda
            temp[0] = n + vec_lambda_ring[0]
            mC_inv_trace = np.sum(1. / temp(temp != 0))
            err_bd = self.uncert * np.sqrt(DSC * RKHS_norm / mC_inv_trace)
        else:
            # empirical Bayes
            if self.avoid_cancel_error:
                DSC = abs(vec_lambda_ring[0] / (n + vec_lambda_ring[0]))
            else:
                DSC = abs(1 - (n / vec_lambda[0]))
            err_bd = self.uncert * np.sqrt(DSC * RKHS_norm / n)

        if self.arb_mean:  # zero mean case
            muhat = ftilde[0] / n
        else:  # non zero mean case
            muhat = ftilde[0] / vec_lambda[0]

        self.error_bound = err_bd
        muhat = np.abs(muhat)
        return muhat, err_bd
    
    # objective function to estimate parameter theta
    # MLE : Maximum likelihood estimation
    # GCV : Generalized cross validation
    def objective_function(self, theta, xun, ftilde):
        n = len(ftilde)
        fudge = 100 * np.finfo(float).eps
        # if type(theta) != np.ndarray:
        #     theta = np.ones((1, xun.shape[1])) * theta
        [vec_lambda, vec_lambda_ring, lambda_factor] = self.kernel(xun, self.order, theta, self.avoid_cancel_error,
                                                                   self.kernType, self.debug_enable)
        vec_lambda = abs(vec_lambda)
        # compute RKHS_norm
        temp = abs(ftilde[vec_lambda > fudge] ** 2) / (vec_lambda[vec_lambda > fudge])

        # compute loss
        if self.errbd_type == 'GCV':
            # GCV
            temp_gcv = abs(ftilde[vec_lambda > fudge] / (vec_lambda[vec_lambda > fudge])) ** 2
            loss1 = 2 * np.log(sum(1. / vec_lambda[vec_lambda > fudge]))
            loss2 = np.log(sum(temp_gcv[1:]))
            # ignore all zero eigenvalues
            loss = loss2 - loss1

            if self.arb_mean:
                RKHS_norm = (1 / lambda_factor) * sum(temp_gcv[1:]) / n
            else:
                RKHS_norm = (1 / lambda_factor) * sum(temp_gcv) / n
        else:
            # default: MLE
            if self.arb_mean:
                RKHS_norm = (1 / lambda_factor) * sum(temp[1:]) / n
                temp_1 = (1 / lambda_factor) * sum(temp[1:])
            else:
                RKHS_norm = (1 / lambda_factor) * sum(temp) / n
                temp_1 = (1 / lambda_factor) * sum(temp)

            # ignore all zero eigenvalues
            loss1 = sum(np.log(abs(lambda_factor * vec_lambda[vec_lambda > fudge])))
            if temp_1 != 0:
                loss2 = n * np.log(temp_1)
            else:
                loss2 = n * np.log(temp_1 + np.finfo(float).eps)
            loss = loss1 + loss2

        if self.debug_enable:
            self.alert_msg(loss1, 'Inf', 'Imag')
            self.alert_msg(RKHS_norm, 'Imag')
            self.alert_msg(loss2, 'Inf', 'Imag')
            self.alert_msg(loss, 'Inf', 'Imag', 'Nan')
            self.alert_msg(vec_lambda, 'Imag')

        vec_lambda, vec_lambda_ring = lambda_factor * vec_lambda, lambda_factor * vec_lambda_ring
        return loss, vec_lambda, vec_lambda_ring, RKHS_norm

    # Computes modified kernel Km1 = K - 1
    # Useful to avoid cancellation error in the computation of (1 - n/\lambda_1)
    @staticmethod
    def kernel_t(aconst, Bern):
        d = np.size(Bern, 1)
        if type(aconst) != np.ndarray:
            theta = np.ones((d, 1)) * aconst
        else:
            theta = aconst  # theta varies per dimension

        Kjm1 = theta[0] * Bern[:, 0]  # Kernel at j-dim minus One
        Kj = 1 + Kjm1  # Kernel at j-dim

        for j in range(1, d):
            Kjm1_prev = Kjm1
            Kj_prev = Kj  # save the Kernel at the prev dim

            Kjm1 = theta[j] * Bern[:, j] * Kj_prev + Kjm1_prev
            Kj = 1 + Kjm1

        Km1 = Kjm1
        K = Kj
        return [Km1, K]

    # prints debug message if the given variable is Inf, Nan or complex, etc
    # Example: alertMsg(x, 'Inf', 'Imag')
    #          prints if variable 'x' is either Infinite or Imaginary
    @staticmethod
    def alert_msg(*args):
        varargin = args
        nargin = len(varargin)
        if nargin > 1:
            i_start = 0
            var_tocheck = varargin[i_start]
            i_start = i_start + 1
            inpvarname = 'variable'

            while i_start < nargin:
                var_type = varargin[i_start]
                i_start = i_start + 1

                if var_type == 'Nan':
                    if np.any(np.isnan(var_tocheck)):
                        print('%s has NaN values' % inpvarname)
                elif var_type == 'Inf':
                    if np.any(np.isinf(var_tocheck)):
                        print('%s has Inf values' % inpvarname)
                elif var_type == 'Imag':
                    if not np.all(np.isreal(var_tocheck)):
                        print('%s has complex values' % inpvarname)
                else:
                    print('unknown type check requested !')
    
    def integrate(self):
        t_start = time()
        data = Data(
            parameters = [
                'solution',
                'comb_bound_low',
                'comb_bound_high',
                'comb_bound_diff',
                'comb_flags',
                'n_total',
                'n',
                'time_integrate'])
        data.flags_indv = np.tile(False,self.integrand.d_indv)
        data.compute_flags = np.tile(True,self.integrand.d_indv)
        data.n = np.tile(self.n_init,self.integrand.d_indv)
        data.n_min = 0 
        data.n_max = self.n_init
        data.solution_indv = np.tile(np.nan,self.integrand.d_indv)
        data.xfull = np.empty((0,self.integrand.d))
        data.yfull = np.empty(self.integrand.d_indv+(0,))
        data.bounds_half_width = np.tile(np.inf,self.integrand.d_indv)
        data.muhat = np.tile(np.nan,self.integrand.d_indv)
        while True:
            m = int(np.log2(data.n_max))
            xnext = self.discrete_distrib(n_min=data.n_min,n_max=data.n_max)
            data.xfull = np.concatenate([data.xfull,xnext],0)
            ynext = self.integrand.f(xnext,periodization_transform=self.ptransform,compute_flags=data.compute_flags)
            ynext[~data.compute_flags] = np.nan
            data.yfull = np.concatenate([data.yfull,ynext],-1)
            if data.n_min==0: # first iteration
                ytildefull = self.ft(ynext)/np.sqrt(2**m)
            else: # any iteration after the first
                mnext = int(m-1)
                ytildeomega = self.omega(mnext)*self.ft(ynext[data.compute_flags])/np.sqrt(2**mnext)
                ytildefull_next = np.nan*np.ones_like(ytildefull)
                ytildefull_next[data.compute_flags] = (ytildefull[data.compute_flags]-ytildeomega)/2
                ytildefull[data.compute_flags] = (ytildefull[data.compute_flags]+ytildeomega)/2
                ytildefull = np.concatenate([ytildefull,ytildefull_next],axis=-1)
            for j in np.ndindex(self.integrand.d_indv):
                if not data.compute_flags[j]: continue 
                data.muhat[j],data.bounds_half_width[j] = self._stopping_criterion(data.xfull,2**m*ytildefull[j],m)
            data.indv_bound_low = data.muhat-data.bounds_half_width
            data.indv_bound_high = data.muhat+data.bounds_half_width
            data.n[data.compute_flags] = data.n_max
            data.n_total = data.n_max
            data.comb_bound_low,data.comb_bound_high = self.integrand.bound_fun(data.indv_bound_low,data.indv_bound_high)
            data.comb_bound_diff = data.comb_bound_high-data.comb_bound_low
            fidxs = np.isfinite(data.comb_bound_low)&np.isfinite(data.comb_bound_high)
            slow,shigh,abs_tols,rel_tols = data.comb_bound_low[fidxs],data.comb_bound_high[fidxs],self.abs_tols[fidxs],self.rel_tols[fidxs]
            data.solution = np.tile(np.nan,data.comb_bound_low.shape)
            data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            data.comb_flags = np.tile(False,data.comb_bound_low.shape)
            data.comb_flags[fidxs] = (shigh-slow) <= (self.error_fun(slow,abs_tols,rel_tols)+self.error_fun(shigh,abs_tols,rel_tols))
            data.flags_indv = self.integrand.dependency(data.comb_flags)
            data.compute_flags = ~data.flags_indv
            if np.sum(data.compute_flags)==0:
                break # sufficiently estimated
            elif 2*data.n_total>self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceeds n_limit = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied. """ \
                % (int(data.n_total),int(data.n_total),int(self.n_limit))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            data.n_min = data.n_max
            data.n_max = 2*data.n_min
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        assert rmse_tol is None, "rmse_tol not supported by this stopping criterion."
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb,self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb,self.rel_tol)
