from ._stopping_criterion import StoppingCriterion
from ..accumulate_data.ld_transform_bayes_data import LDTransformBayesData
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning, NotYetImplemented
import numpy as np
from time import time
import warnings


class _CubBayesLDG(StoppingCriterion):
    """
    Abstract class for CubBayes{LD}G where LD is a low discrepancy discrete distribution.
    See subclasses for implementation differences for each LD sequence.
    """

    def __init__(self, integrand, fbt, merge_fbt, ptransform, allowed_distribs, kernel,
                 abs_tol, rel_tol, n_init, n_max, alpha, error_fun):

        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        m_min = np.log2(n_init)
        m_max = np.log2(n_max)
        if m_min % 1 != 0. or m_min < 5 or m_max % 1 != 0:
            warning_s = '''
                n_init and n_max must be a powers of 2.
                n_init must be >= 2^8.
                Using n_init = 2^8 and n_max=2^22.'''
            warnings.warn(warning_s, ParameterWarning)
            m_min = 8.
            m_max = 22.
        self.m_min = m_min
        self.m_max = m_max
        self.n_init = n_init  # number of samples to start with = 2^mmin
        self.n_max = n_max  # max number of samples allowed = 2^mmax
        self.alpha = alpha  # p-value, default 0.1%.
        self.stop_at_tol = True  # automatic mode: stop after meeting the error tolerance
        self.arb_mean = True  # by default use zero mean algorithm
        self.avoid_cancel_error = True  # avoid cancellation error in stopping criterion
        self.debug_enable = False  # enable debug prints
        self.data = None
        self.fbt, self.merge_fbt = fbt, merge_fbt
        self.ptransform = ptransform  # periodization transform
        self.kernel = kernel

        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib

        # Sobol indices
        self.dprime = self.integrand.dprime
        self.cv = []
        self.ncv = len(self.cv)
        self.cast_complex = False
        self.d = self.discrete_distrib.d
        self.error_fun = error_fun

        # Verify Compliant Construction
        allowed_levels = ['single']
        allow_vectorized_integrals = True
        super(_CubBayesLDG, self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

    def integrate(self):
        t_start = time()
        self.datum = np.empty(self.dprime, dtype=object)
        for j in np.ndindex(self.dprime):
            self.datum[j] = LDTransformBayesData(self, self.integrand, self.true_measure, self.discrete_distrib,
                                                 self.m_min, self.m_max, self.fbt, self.merge_fbt, self.kernel)

        self.data = LDTransformBayesData.__new__(LDTransformBayesData)
        self.data.flags_indv = np.tile(True, self.dprime)
        prev_flags_indv = self.data.flags_indv
        self.data.m = np.tile(self.m_min, self.dprime)
        self.data.n_min = 0
        self.data.ci_low = np.tile(-np.inf, self.dprime)
        self.data.ci_high = np.tile(np.inf, self.dprime)
        self.data.solution_indv = np.tile(np.nan, self.dprime)
        self.data.solution = np.nan
        self.data.xfull = np.empty((0, self.d))
        self.data.yfull = np.empty((0,) + self.dprime)
        stop_flag = np.tile(None, self.dprime)
        while True:
            m = self.data.m.max()
            n_min = self.data.n_min
            n_max = int(2 ** m)
            n = int(n_max - n_min)
            xnext, xnext_un = self.discrete_distrib.gen_samples(n_min=n_min, n_max=n_max, return_unrandomized=True,
                                                                warn=False)
            ycvnext = np.empty((1 + self.ncv, n,) + self.dprime, dtype=float)
            ycvnext[0] = self.integrand.f(xnext, periodization_transform=self.ptransform,
                                          compute_flags=self.data.flags_indv)
            for k in range(self.ncv):
                ycvnext[1 + k] = self.cv[k].f(xnext, periodization_transform=self.ptransform,
                                              compute_flags=self.data.flags_indv)
            # print(self.data.flags_indv)
            for j in np.ndindex(self.dprime):
                if prev_flags_indv[j] == False and self.data.flags_indv[j] == True:
                    raise NotYetImplemented('_CubBayesLDG: Cannot resume data update: flags_indv[j] switched from False to True')
                if not self.data.flags_indv[j]:
                    continue
                slice_yj = (0, slice(None),) + j
                if type(self.discrete_distrib).__name__ == 'DigitalNetB2':
                    y_val = ycvnext[slice_yj].copy()  # to statisfy C_CONTIGUOUS
                else:
                    y_val = ycvnext[slice_yj]

                # Update function values
                success, muhat, r_order, err_bd, _ = self.datum[j].update_data(y_val_new=y_val, xnew=xnext, xunnew=xnext_un)

                bounds = muhat + np.array([-1, 1]) * err_bd
                stop_flag[j], self.data.solution_indv[j], self.data.ci_low[j], self.data.ci_high[j] = \
                    success, muhat, bounds[0], bounds[1]

            self.data.xfull = np.vstack((self.data.xfull, xnext))
            self.data.yfull = np.vstack((self.data.yfull, ycvnext[0]))
            self.data.indv_error = (self.data.ci_high - self.data.ci_low) / 2
            self.data.ci_comb_low, self.data.ci_comb_high, self.data.violated = self.integrand.bound_fun(
                self.data.ci_low, self.data.ci_high)
            error_low = self.error_fun(self.data.ci_comb_low, self.abs_tol, self.rel_tol)
            error_high = self.error_fun(self.data.ci_comb_high, self.abs_tol, self.rel_tol)
            self.data.solution = 1 / 2 * (self.data.ci_comb_low + self.data.ci_comb_high + error_low - error_high)
            rem_error_low = abs(self.data.ci_comb_low - self.data.solution) - error_low
            rem_error_high = abs(self.data.ci_comb_high - self.data.solution) - error_high
            self.data.flags_comb = np.maximum(rem_error_low, rem_error_high) >= 0
            self.data.flags_comb |= self.data.violated
            prev_flags_indv = self.data.flags_indv
            self.data.flags_indv = self.integrand.dependency(self.data.flags_comb)
            self.data.n = 2 ** self.data.m
            self.data.n_total = self.data.n.max()

            if np.sum(self.data.flags_indv) == 0:
                break  # stopping criterion met
            elif 2 * self.data.n_total > self.n_max:
                # doubling samples would go over n_max
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceed n_max = %d.
                No more samples will be generated.
                Note that error tolerances may no longer be satisfied.""" \
                            % (int(self.data.n_total), int(self.data.n_total), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                self.data.n_min = n_max
                self.data.m += self.data.flags_indv

        self.data.integrand = self.integrand
        self.data.true_measure = self.true_measure
        self.data.discrete_distrib = self.discrete_distrib
        self.data.stopping_crit = self
        self.data.parameters = [
            'solution',
            'indv_error',
            'ci_low',
            'ci_high',
            'ci_comb_low',
            'ci_comb_high',
            'flags_comb',
            'flags_indv',
            'n_total',
            'n',
            'time_integrate']
        self.data.datum = self.datum
        self.data.time_integrate = time() - t_start
        return self.data.solution, self.data

    # computes the integral
    # ## Obsolete - do not use ##
    def integrate_1d(self):
        # Construct AccumulateData Object to House Integration data
        self.data = LDTransformBayesData(self, self.integrand, self.true_measure, self.discrete_distrib,
                                         self.m_min, self.m_max, self.fbt, self.merge_fbt, self.kernel)
        tstart = time()  # start the timer

        # Iteratively find the number of points required for the cubature to meet
        # the error threshold
        while True:
            # Update function values
            stop_flag, muhat, order_, err_bnd, m = self.data.update_data()

            # if stop_at_tol true, exit the loop
            # else, run for for all 'n' values.
            # Used to compute error values for 'n' vs error plotting
            if self.stop_at_tol and stop_flag:
                break

            if m >= self.m_max:
                warnings.warn('''
                    Already used maximum allowed sample size %d.
                    Note that error tolerances may no longer be satisfied.''' % (2 ** self.m_max),
                              MaxSamplesWarning)
                break

        self.data.time_integrate = time() - tstart
        # Approximate integral
        self.data.solution = muhat

        return muhat, self.data
