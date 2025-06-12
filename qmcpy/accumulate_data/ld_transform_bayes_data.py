from ._accumulate_data import AccumulateData
from ..util import MaxSamplesWarning
from ..discrete_distribution import Lattice
import warnings
import numpy as np
from scipy.optimize import fminbound as fminbnd
from scipy.optimize import fmin, fmin_bfgs
from scipy.stats import norm as gaussnorm
from scipy.stats import t as tnorm


class LDTransformBayesData(AccumulateData):
    """
    Update and store transformation data based on low-discrepancy sequences.
    See the stopping criterion that utilize this object for references.
    """

    def __init__(self, stopping_crit, integrand, true_measure, discrete_distrib, m_min: int, m_max: int,
                 fbt, merge_fbt, kernel, alpha):
        """
        Args:
            stopping_crit (AbstractStoppingCriterion): a AbstractStoppingCriterion instance
            integrand (AbstractIntegrand): an AbstractIntegrand instance
            true_measure (AbstractTrueMeasure): A AbstractTrueMeasure instance
            discrete_distrib (AbstractDiscreteDistribution): an AbstractDiscreteDistribution instance
            m_min (int): initial n == 2^m_min
            m_max (int): max n == 2^m_max
        """
        self.parameters = ['solution', 'error_bound', 'n_total']
        self.stopping_crit = stopping_crit
        self.integrand = integrand
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        self.distribution_name = type(self.discrete_distrib).__name__

        # Bayes cubature properties
        self.errbd_type = self.stopping_crit.errbd_type
        self.arb_mean = self.stopping_crit.arb_mean
        self.order = self.stopping_crit.order
        self.kernType = self.stopping_crit.kernType
        self.avoid_cancel_error = self.stopping_crit.avoid_cancel_error
        self.abs_tol = self.stopping_crit.abs_tol
        self.rel_tol = self.stopping_crit.rel_tol
        self.debug_enable = self.stopping_crit.debug_enable

        # Credible interval : two-sided confidence, i.e., 1-alpha percent quantile
        # quantile value for the error bound
        if self.errbd_type == 'FULL':
            # degrees of freedom = 2^mmin - 1
            self.uncert = -tnorm.ppf(alpha / 2, (2 ** m_min) - 1)
        else:
            self.uncert = -gaussnorm.ppf(alpha / 2)

        # Set Attributes
        self.m_min = m_min
        self.m_max = m_max
        self.debugEnable = True

        self.n_total = 0  # total number of samples generated
        self.solution = np.nan

        self.iter = 0
        self.m = self.m_min
        self.mvec = np.arange(self.m_min, self.m_max + 1, dtype=int)

        # Initialize various temporary storage between iterations
        self.xpts_ = np.array([])  # shifted lattice points
        self.xun_ = np.array([])  # un-shifted lattice points
        self.ftilde_ = np.array([])  # fourier transformed integrand values
        if isinstance(self.discrete_distrib, Lattice):
            # integrand after the periodization transform
            self.ff = lambda x, *args, **kwargs: self.integrand.f(x,
                                                                  periodization_transform=stopping_crit.ptransform,
                                                                  *args, **kwargs).squeeze()
        else:
            self.ff = self.integrand.f
        self.fbt = fbt
        self.merge_fbt = merge_fbt
        self.kernel = kernel

        super(LDTransformBayesData, self).__init__()

    def update_data(self, y_val_new=None, xnew=None, xunnew=None):
        """ See abstract method. """
        # Generate sample values

        if self.iter < len(self.mvec):
            if y_val_new is None:
                self.ftilde_, self.xun_, self.xpts_ = self.iter_fbt(self.iter, self.xun_, self.xpts_, self.ftilde_)
            else:
                self.ftilde_, self.xun_, self.xpts_ = self.iter_fbt(self.iter, self.xun_, self.xpts_, self.ftilde_,
                                                                    y_val_new, xunnew, xnew)
            self.m = self.mvec[self.iter]
            self.iter += 1
            # update total samples
            self.n_total = 2 ** self.m  # updated the total evaluations
        else:
            warnings.warn('''
                Already used maximum allowed sample size %d.
                Note that error tolerances may no longer be satisfied.''' % (2 ** self.m_max),
                          MaxSamplesWarning)

        return self._stopping_criterion(self.xun_, self.ftilde_, self.m)

    # decides if the user-defined error threshold is met
    def _stopping_criterion(self, xpts, ftilde, m):
        r = self.stopping_crit.order
        ftilde = ftilde.squeeze()
        n = 2 ** m
        success = False
        lna_range = [-5, 0]  # reduced from [-5, 5], to avoid kernel values getting too big causing error

        # search for optimal shape parameter
        if self.stopping_crit.one_theta == True:
            lna_MLE = fminbnd(lambda lna: self.objective_function(np.exp(lna), xpts, ftilde)[0],
                              x1=lna_range[0], x2=lna_range[1], xtol=1e-2, disp=0)

            aMLE = np.exp(lna_MLE)
            _, vec_lambda, vec_lambda_ring, RKHS_norm = self.objective_function(aMLE, xpts, ftilde)
        else:
            if self.stopping_crit.use_gradient == True:
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
        muminus = muhat - err_bd
        muplus = muhat + err_bd

        if 2 * err_bd <= max(self.abs_tol, self.rel_tol * abs(muminus)) + max(self.abs_tol, self.rel_tol * abs(muplus)):
            if err_bd == 0:
                err_bd = np.finfo(float).eps

            # stopping criterion achieved
            success = True
        return success, muhat, r, err_bd, m

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
