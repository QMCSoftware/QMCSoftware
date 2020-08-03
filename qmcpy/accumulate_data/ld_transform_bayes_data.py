from ._accumulate_data import AccumulateData
from ..util import CubatureWarning, MaxSamplesWarning

from numpy import array, nan, zeros, tile, inf, hstack, arange, where
import warnings
import numpy as np

class OutOptParams():
    def __init__(self):
        self.ErrBdAll = array([])
        self.muhatAll = array([])
        self.mvec = array([])
        self.aMLEAll = array([])
        self.timeAll = array([])
        self.s_All = array([])
        self.dscAll = array([])
        self.absTol = None
        self.relTol = None
        self.shift = None
        self.stopAtTol = None
        self.r = None

class OutParams():
    def __init__(self):
        self.n = None
        self.time = None
        self.ErrBd = None
        self.optParams = OutOptParams()
        self.exitflag = None

class LDTransformBayesData(AccumulateData):

    parameters = ['n_total','solution','r_lag']

    def __init__(self, stopping_criterion, integrand, m_min, m_max, shift, vdc_order):
        """
        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            basis_transform (method): Transform ynext, combine with y, and then transform all points. 
                For cub_lattice this is Fast Fourier Transform (FFT). 
                For cub_sobol this is Fast Walsh Transform (FWT)
            m_min (int): initial n == 2^m_min
            m_max (int): max n == 2^m_max
            shift (function): random shift for the lattice points
            vdc_order (boolean): if True, lattice points used in VDC order else in linear order
        """
        # Extract attributes from integrand
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        self.shift = shift
        self.vdc_order = vdc_order
        self.dim = stopping_criterion.dim

        #  generator for the Lattice points
        self.gen_vec = self.get_lattice_gen_vec(stopping_criterion.dim)

        # Set Attributes
        self.m_min = m_min
        self.m_max = m_max
        self.n_total = 0  # total number of samples generated
        self.solution = nan
        self.debugEnable = True

        self.iter = 0
        self.m = self.m_min
        self.mvec = np.arange(self.m_min, self.m_max+1, dtype=int)

        # Initialize various temporary storage between iterations
        self.xpts_ = array([])  # shifted lattice points
        self.xun_ = array([])  # unshifted lattice points
        self.ftilde_ = array([])  # fourier transformed integrand values
        self.ff = self.doPeriodTx(self.integrand.f, stopping_criterion.ptransform)  # integrand after the periodization transform

        super(LDTransformBayesData,self).__init__()

    def update_data(self):
        """ See abstract method. """
        # Generate sample values

        if self.iter < len(self.mvec):
            if self.vdc_order:
                self.ftilde_, self.xun_, self.xpts_ = self.iter_fft_vdc(self.iter, self.xun_, self.xpts_, self.ftilde_)
            else:
                self.ftilde_, self.xun_, self.xpts_ = self.iter_fft(self.iter, self.xun_, self.xpts_, self.ftilde_)

            self.m = self.mvec[self.iter]
            self.iter += 1
        else:
            warnings.warn(f'Already used maximum allowed sample size {2**self.m_max}.'
                          f' Note that error tolerances may no longer be satisfied',
                          MaxSamplesWarning)

        return self.xun_, self.ftilde_, self.m

    # Efficient FFT computation algorithm, avoids recomputing the full fft
    def iter_fft(self, iter, xun, xpts, ftildePrev):
        m = self.mvec[iter]
        n = 2 ** m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FFT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 0:
            # In the first iteration compute full FFT
            # xun_ = mod(bsxfun( @ times, (0:1 / n:1-1 / n)',self.gen_vec),1)
            xun_ = np.arange(0, 1, 1 / n).reshape((n, 1))
            xun_ = np.mod((xun_ * self.gen_vec), 1)

            xun_ = self.distribution.gen_samples(n_min=0, n_max=n, warn=False)

            # xpts_ = np.mod(bsxfun( @ plus, xun_, shift), 1)  # shifted
            xpts_ = np.mod((xun_ + self.shift), 1)  # shifted

            # Compute initial FFT
            ftilde_ = np.fft.fft(self.ff(xpts_))  # evaluate integrand's fft
            ftilde_ = ftilde_.reshape((n, 1))
        else:
            # xunnew = np.mod(bsxfun( @ times, (1/n : 2/n : 1-1/n)',self.gen_vec),1)
            xunnew = np.arange(1 / n, 1, 2 / n).reshape((n//2, 1))
            xunnew = np.mod(xunnew * self.gen_vec, 1)

            xunnew = self.distribution.gen_samples(n_min=n//2, n_max=n)

            # xnew = np.mod(bsxfun( @ plus, xunnew, shift), 1)
            xnew = np.mod((xunnew + self.shift), 1)

            [xun_, xpts_] = self.merge_pts(xun, xunnew, xpts, xnew, n, self.dim)
            mnext = m - 1

            # Compute FFT on next set of new points
            ftildeNextNew = np.fft.fft(self.ff(xnew))
            ftildeNextNew = ftildeNextNew.reshape((n//2, 1))
            if self.debugEnable:
                self.alertMsg(ftildeNextNew, 'Nan', 'Inf')

            # combine the previous batch and new batch to get FFT on all points
            ftilde_ = self.merge_fft(ftildePrev, ftildeNextNew, mnext)

        return ftilde_, xun_, xpts_

    # Lattice points are ordered in van der Corput sequence, so we cannot use
    # Matlab's built-in fft routine. We use a custom fft instead.
    def iter_fft_vdc(self, iter, xun, xpts, ftildePrev):
        m = self.mvec[iter]
        n = 2 ** m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FFT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 1:
            # in the first iteration compute the full FFT
            [xpts_, xun_] = self.simple_lattice_gen(n, self.dim, self.shift, True)

            # Compute initial FFT
            ftilde_ = self.fft_DIT(self.ff(xpts_), m)  # evaluate integrand's fft
        else:
            [xnew, xunnew] = self.simple_lattice_gen(n, self.dim, self.shift, False)
            mnext = m - 1

            # Compute FFT on next set of new points
            ftildeNextNew = self.fft_DIT(self.ff(xnew), mnext)
            if self.debugEnable:
                self.alertMsg(ftildeNextNew, 'Nan', 'Inf')

            xpts_ = np.vstack(xpts, xnew)
            temp = np.zeros(n, self.dim)
            temp[0:: 2, :] = xun
            temp[1:: 2, :] = xunnew
            xun_ = temp
            # combine the previous batch and new batch to get FFT on all points
            ftilde_ = self.merge_fft(ftildePrev, ftildeNextNew, mnext)

        if self.debugEnable:
            self.alertMsg(ftilde_, 'Inf', 'Nan')

        return ftilde_, xun_, xpts_


    # using FFT butefly plot technique merges two halves of fft
    @staticmethod
    def merge_fft(ftildeNew, ftildeNextNew, mnext):
        ftildeNew = np.vstack([ftildeNew, ftildeNextNew])
        nl = 2 ** mnext
        # ptind=[true(nl,1); false(nl,1)]
        ptind = np.ndarray(shape=(2 * nl, 1), buffer=np.array([True] * nl + [False] * nl), dtype=bool)
        # coef = exp(-2*1j*(0:nl-1)'/(2*nl))
        coef = np.exp(-2 * np.pi * 1j * np.ndarray(shape=(nl, 1), buffer=np.arange(0, nl), dtype=int) / (2 * nl))
        # coefv = np.matlib.repmat(coef, 1, 1)
        coefv = np.tile(coef, (1, 1))
        evenval = ftildeNew[ptind].reshape((nl, 1))
        oddval = ftildeNew[~ptind].reshape((nl, 1))
        ftildeNew[ptind] = np.squeeze(evenval + coefv * oddval)
        ftildeNew[~ptind] = np.squeeze(evenval - coefv * oddval)
        return ftildeNew

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


    # just returns the generator for rank-1 Lattice point generation
    @staticmethod
    def get_lattice_gen_vec(d):
        z = [1, 433461, 315689, 441789, 501101, 146355, 88411, 215837, 273599,
             151719, 258185, 357967, 96407, 203741, 211709, 135719, 100779,
             85729, 14597, 94813, 422013, 484367]  # generator
        z = np.array(z[:d]).reshape((1, d))
        return z

    # generates rank-1 Lattice points in Vander Corput sequence order
    @staticmethod
    def simple_lattice_gen(n, d, shift, firstBatch):
        z = LDTransformBayesData.get_lattice_gen_vec(d)

        nmax = n
        nmin = 1 + n / 2
        if firstBatch == True:
            nmin = 1

        nelem = nmax - nmin + 1

        if firstBatch == True:
            brIndices = LDTransformBayesData.vdc(nelem).transpose()

            # xlat_ = np.mod(bsxfun( @ times, (0:1 / n:1-1 / n)',z),1) # unshifted in direct order
            xlat_ = np.mod((np.arange(0, 1 - 1 / n, 1 / n).transpose() * z), 1)
        else:
            brIndices = LDTransformBayesData.vdc(nelem).transpose() + 1 / (2 * (nmin - 1))
            # xlat_ = mod(bsxfun( @ times, (1 / n:2 / n:1-1 / n)',z),1) # unshifted in direct order
            xlat_ = np.mod((np.arange(1 / n, 1 - 1 / n, 2 / n).transpose() * z), 1)

        # xlat = mod(bsxfun( @ times, brIndices',z),1)  # unshifted
        xlat = np.mod((brIndices.transpose() * z), 1)
        # xlat = mod(bsxfun( @ plus, xlat, shift), 1)  # shifted in VDC order
        xlat = np.mod((xlat + shift), 1)

        return [xlat, xlat_]

    # van der Corput sequence in base 2
    @staticmethod
    def vdc(n):
        if n > 1:
            k = np.log2(n)  # We compute the VDC seq part by part of power 2 size
            q = np.zeros(2 ** k, 1)
            for l in range(0, k - 1):
                nl = 2 ** l
                kk = 2 ** (k - l - 1)
                ptind = np.matlib.repmat(np.vstack(np.full((nl, 1), False), np.full((nl, 1), True)), kk, 1)
                q[ptind] = q[ptind] + 1 / 2 ** (l + 1)
        else:
            q = 0
        return q

    # fft with decimation in time i.e., input is already in 'bitrevorder'
    @staticmethod
    def fft_DIT(y, nmmin):
        for l in range(0, nmmin - 1):
            nl = 2 ** l
            nmminlm1 = 2 ** (nmmin - l - 1)
            # ptind=repmat([true(nl,1); false(nl,1)], nmminlm1,1)
            ptind = np.matlib.repmat(np.vstack(np.full((nl, 1), True), np.full((nl, 1), False)), nmminlm1, 1)
            # coef=exp(-2*pi()*sqrt(-1)*(0:nl-1)'/(2*nl))
            coef = np.exp(-2 * np.pi * 1j * np.arange(0, nl - 1).transpose() / (2 * nl))
            coefv = np.matlib.repmat(coef, nmminlm1, 1)
            evenval = y[ptind]
            oddval = y[~ptind]
            y[ptind] = (evenval + coefv * oddval)
            y[~ptind] = (evenval - coefv * oddval)
        return y


    # computes the periodization transform for the given function values
    @staticmethod
    def doPeriodTx(fInput, ptransform):

        if ptransform == 'Baker':
            f = lambda x: fInput(1 - 2 * abs(x - 1 / 2))  # Baker's transform
        elif ptransform == 'C0':
            f = lambda x: fInput(3 * x ** 2 - 2 * x ** 3) * np.prod(6 * x * (1 - x), 1)  # C^0 transform
        elif ptransform == 'C1':
            # C^1 transform
            f = lambda x: fInput(x ** 3 * (10 - 15 * x + 6 * x ** 2)) * np.prod(30 * x ** 2 * (1 - x) ** 2, 1)
        elif ptransform == 'C1sin':
            # Sidi C^1 transform
            f = lambda x: fInput(x - np.sin(2 * np.pi * x) / (2 * np.pi)) * np.prod(2 * np.sin(np.pi * x) ** 2, 1)
        elif ptransform == 'C2sin':
            # Sidi C^2 transform
            psi3 = lambda t: (8 - 9 * np.cos(np.pi * t) + np.cos(3 * np.pi * t)) / 16
            psi3_1 = lambda t: (9 * np.sin(np.pi * t) * np.pi - np.sin(3 * np.pi * t) * 3 * np.pi) / 16
            f = lambda x: fInput(psi3(x)) * np.prod(psi3_1(x), 1)
        elif ptransform == 'C3sin':
            # Sidi C^3 transform
            psi4 = lambda t: (12 * np.pi * t - 8 * np.sin(2 * np.pi * t) + np.sin(4 * np.pi * t)) / (12 * np.pi)
            psi4_1 = lambda t: (12 * np.pi - 8 * np.cos(2 * np.pi * t) * 2 * np.pi + np.sin(
                4 * np.pi * t) * 4 * np.pi) / (12 * np.pi)
            f = lambda x: fInput(psi4(x)) * np.prod(psi4_1(x), 1)
        elif ptransform == 'none':
            # do nothing
            f = lambda x: fInput(x)
        else:
            f = fInput
            print(f'Error: Periodization transform {ptransform} not implemented')

        return f

    # prints debug message if the given variable is Inf, Nan or complex, etc
    # Example: alertMsg(x, 'Inf', 'Imag')
    #          prints if variable 'x' is either Infinite or Imaginary
    @staticmethod
    def alertMsg(*args):
        varargin = args
        nargin = len(varargin)
        if nargin > 1:
            iStart = 0
            varTocheck = varargin[iStart]
            iStart = iStart + 1
            inpvarname = 'variable'

            while iStart < nargin:
                type = varargin[iStart]
                iStart = iStart + 1

                if type == 'Nan':
                    if np.any(np.isnan(varTocheck)):
                        print(f'{inpvarname} has NaN values')
                elif type == 'Inf':
                    if np.any(np.isinf(varTocheck)):
                        print(f'{inpvarname} has Inf values')
                elif type == 'Imag':
                    if not np.all(np.isreal(varTocheck)):
                        print(f'{inpvarname} has complex values')
                else:
                    print('unknown type check requested !')
