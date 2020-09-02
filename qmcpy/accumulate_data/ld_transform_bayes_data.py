from ._accumulate_data import AccumulateData
from ..integrand import do_period_transform
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

    def __init__(self, stopping_criterion, integrand, m_min: int, m_max: int):
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
        self.ff = do_period_transform(self.integrand.f,
                                           stopping_criterion.ptransform)  # integrand after the periodization transform

        super(LDTransformBayesData, self).__init__()

    def update_data(self):
        """ See abstract method. """
        # Generate sample values

        if self.iter < len(self.mvec):
            self.ftilde_, self.xun_, self.xpts_ = self.iter_fft(self.iter, self.xun_, self.xpts_, self.ftilde_)

            self.m = self.mvec[self.iter]
            self.iter += 1
            # update total samples
            self.n_total = 2 ** self.m  # updated the total evaluations
        else:
            warnings.warn(f'Already used maximum allowed sample size {2 ** self.m_max}.'
                          f' Note that error tolerances may no longer be satisfied',
                          MaxSamplesWarning)

        return self.xun_, self.ftilde_, self.m

    # Efficient FFT computation algorithm, avoids recomputing the full fft
    def iter_fft(self, iter, xun, xpts, ftilde_prev):
        m = self.mvec[iter]
        n = 2 ** m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FFT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 0:
            # In the first iteration compute full FFT
            # xun_ = mod(bsxfun( @ times, (0:1 / n:1-1 / n)',self.gen_vec),1)
            # xun_ = np.arange(0, 1, 1 / n).reshape((n, 1))
            # xun_ = np.mod((xun_ * self.gen_vec), 1)
            # xpts_ = np.mod(bsxfun( @ plus, xun_, shift), 1)  # shifted

            xpts_, xun_ = self.distribution.gen_samples(n_min=0, n_max=n, warn=False, return_non_random=True)

            # Compute initial FFT
            ftilde_ = np.fft.fft(self.ff(xpts_))  # evaluate integrand's fft
            ftilde_ = ftilde_.reshape((n, 1))
        else:
            # xunnew = np.mod(bsxfun( @ times, (1/n : 2/n : 1-1/n)',self.gen_vec),1)
            # xunnew = np.arange(1 / n, 1, 2 / n).reshape((n // 2, 1))
            # xunnew = np.mod(xunnew * self.gen_vec, 1)
            # xnew = np.mod(bsxfun( @ plus, xunnew, shift), 1)

            xnew, xunnew = self.distribution.gen_samples(n_min=n // 2, n_max=n, return_non_random=True)

            [xun_, xpts_] = self.merge_pts(xun, xunnew, xpts, xnew, n, self.dim)
            mnext = m - 1

            # Compute FFT on next set of new points
            ftilde_next_new = np.fft.fft(self.ff(xnew))
            ftilde_next_new = ftilde_next_new.reshape((n // 2, 1))
            if self.debugEnable:
                self.alert_msg(ftilde_next_new, 'Nan', 'Inf')

            # combine the previous batch and new batch to get FFT on all points
            ftilde_ = self.merge_fft(ftilde_prev, ftilde_next_new, mnext)

        return ftilde_, xun_, xpts_

    # using FFT butefly plot technique merges two halves of fft
    @staticmethod
    def merge_fft(ftilde_new, ftilde_next_new, mnext):
        ftilde_new = np.vstack([ftilde_new, ftilde_next_new])
        nl = 2 ** mnext
        ptind = np.ndarray(shape=(2 * nl, 1), buffer=np.array([True] * nl + [False] * nl), dtype=bool)
        coef = np.exp(-2 * np.pi * 1j * np.ndarray(shape=(nl, 1), buffer=np.arange(0, nl), dtype=int) / (2 * nl))
        coefv = np.tile(coef, (1, 1))
        evenval = ftilde_new[ptind].reshape((nl, 1))
        oddval = ftilde_new[~ptind].reshape((nl, 1))
        ftilde_new[ptind] = np.squeeze(evenval + coefv * oddval)
        ftilde_new[~ptind] = np.squeeze(evenval - coefv * oddval)
        return ftilde_new

    # inserts newly generated points with the old set by interleaving them
    # xun - unshifted points
    @staticmethod
    def merge_pts(xun, xunnew, x, xnew, n, d):
        temp = np.zeros((n, d))
        temp[0::2, :] = xun
        temp[1::2, :] = xunnew
        xun = temp
        temp = np.zeros((n, d))
        temp[0::2, :] = x
        temp[1::2, :] = xnew
        x = temp
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
