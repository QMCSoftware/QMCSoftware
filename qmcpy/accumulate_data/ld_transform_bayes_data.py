from ._accumulate_data import AccumulateData
from ..util import MaxSamplesWarning

from numpy import array, nan
import warnings
import numpy as np


class LDTransformBayesData(AccumulateData):
    """
    Update and store transformation data based on low-discrepancy sequences.
    See the stopping criterion that utilize this object for references.
    """

    parameters = ['n_total', 'solution', 'error_bound']

    def __init__(self, stopping_criterion, integrand, m_min: int, m_max: int, fbt, merge_fbt):
        """
        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            m_min (int): initial n == 2^m_min
            m_max (int): max n == 2^m_max
        """
        # Extract attributes from integrand
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        self.distribution_name = type(self.distribution).__name__
        self.dim = stopping_criterion.dim

        # Set Attributes
        self.m_min = m_min
        self.m_max = m_max
        self.debugEnable = True

        self.n_total = 0  # total number of samples generated
        self.solution = nan

        self.iter = 0
        self.m = self.m_min
        self.mvec = np.arange(self.m_min, self.m_max + 1, dtype=int)

        # Initialize various temporary storage between iterations
        self.xpts_ = array([])  # shifted lattice points
        self.xun_ = array([])  # un-shifted lattice points
        self.ftilde_ = array([])  # fourier transformed integrand values
        if self.distribution_name == 'Lattice':
            self.ff = self.integrand.period_transform(
                stopping_criterion.ptransform)  # integrand after the periodization transform
        else:
            self.ff = self.integrand.f
        self.fbt = fbt
        self.merge_fbt = merge_fbt

        super(LDTransformBayesData, self).__init__()

    def update_data(self):
        """ See abstract method. """
        # Generate sample values

        if self.iter < len(self.mvec):
            self.ftilde_, self.xun_, self.xpts_ = self.iter_fbt(self.iter, self.xun_, self.xpts_, self.ftilde_)

            self.m = self.mvec[self.iter]
            self.iter += 1
            # update total samples
            self.n_total = 2 ** self.m  # updated the total evaluations
        else:
            warnings.warn(f'Already used maximum allowed sample size {2 ** self.m_max}.'
                          f' Note that error tolerances may no longer be satisfied',
                          MaxSamplesWarning)

        return self.xun_, self.ftilde_, self.m

    # Efficient Fast Bayesian Transform computation algorithm, avoids recomputing the full transform
    def iter_fbt(self, iter, xun, xpts, ftilde_prev):
        m = self.mvec[iter]
        n = 2 ** m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FBT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 0:
            # In the first iteration compute full FBT
            # xun_ = mod(bsxfun( @ times, (0:1 / n:1-1 / n)',self.gen_vec),1)
            # xun_ = np.arange(0, 1, 1 / n).reshape((n, 1))
            # xun_ = np.mod((xun_ * self.gen_vec), 1)
            # xpts_ = np.mod(bsxfun( @ plus, xun_, shift), 1)  # shifted

            xpts_,xun_ = self.gen_samples(n_min=0, n_max=n, return_unrandomized=True, distribution=self.distribution)

            # Compute initial FBT
            ftilde_ = self.fbt(self.ff(xpts_))
            ftilde_ = ftilde_.reshape((n, 1))
        else:
            # xunnew = np.mod(bsxfun( @ times, (1/n : 2/n : 1-1/n)',self.gen_vec),1)
            # xunnew = np.arange(1 / n, 1, 2 / n).reshape((n // 2, 1))
            # xunnew = np.mod(xunnew * self.gen_vec, 1)
            # xnew = np.mod(bsxfun( @ plus, xunnew, shift), 1)

            xnew, xunnew = self.gen_samples(n_min=n // 2, n_max=n, return_unrandomized=True, distribution=self.distribution)
            [xun_, xpts_] = self.merge_pts(xun, xunnew, xpts, xnew, n, self.dim, distribution=self.distribution_name)
            mnext = m - 1
            ftilde_next_new = self.fbt(self.ff(xnew))

            ftilde_next_new = ftilde_next_new.reshape((n // 2, 1))
            if self.debugEnable:
                self.alert_msg(ftilde_next_new, 'Nan', 'Inf')

            # combine the previous batch and new batch to get FBT on all points
            ftilde_ = self.merge_fbt(ftilde_prev, ftilde_next_new, mnext)

        return ftilde_, xun_, xpts_

    @staticmethod
    def gen_samples(n_min, n_max, return_unrandomized, distribution):
        warn = False if n_min == 0 else True
        if type(distribution).__name__ == 'Lattice':
            xpts_, xun_ = distribution.gen_samples(n_min=n_min, n_max=n_max, warn=warn, return_unrandomized=return_unrandomized)
        else:
            xpts_, xun_ = distribution.gen_samples(n_min=n_min, n_max=n_max, warn=warn, return_jlms=return_unrandomized)
        return xpts_, xun_

    # inserts newly generated points with the old set by interleaving them
    # xun - unshifted points
    @staticmethod
    def merge_pts(xun, xunnew, x, xnew, n, d, distribution):
        if distribution == 'Lattice':
            temp = np.zeros((n, d))
            temp[0::2, :] = xun
            temp[1::2, :] = xunnew
            xun = temp
            temp = np.zeros((n, d))
            temp[0::2, :] = x
            temp[1::2, :] = xnew
            x = temp
        else:
            x = np.vstack([x, xnew])
            xun = np.vstack([xun, xunnew])
        return xun, x

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
                        print(f'{inpvarname} has NaN values')
                elif var_type == 'Inf':
                    if np.any(np.isinf(var_tocheck)):
                        print(f'{inpvarname} has Inf values')
                elif var_type == 'Imag':
                    if not np.all(np.isreal(var_tocheck)):
                        print(f'{inpvarname} has complex values')
                else:
                    print('unknown type check requested !')
