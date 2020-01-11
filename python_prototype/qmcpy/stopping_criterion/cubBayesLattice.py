""" Definition for CubBayesLattice_g, a concrete implementation of StoppingCriterion """
import numpy as np
from scipy.optimize import fminbound as fminbnd
from numpy import sqrt, exp, log
from ._stopping_criterion import StoppingCriterion


class CubBayesLattice_g(StoppingCriterion):
    """
    Stopping criterion for Bayesian Cubature using Lattice sequence with guaranteed accuracy
    """

    def __init__(self, abs_tol=1e-2, rel_tol=0,
                 n_init=1024, n_max=1e10, alpha=0.01):
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_max = n_max
        self.alpha = alpha

    def ObjectiveFunction(self, a, xpts, ftilde):
        pass

    def CheckGaussianDensity(self, ftilde, Lambda):
        pass

    # decides if the user-defined error threshold is met
    # function [success,muhat] =
    def stopping_criterion(self, xpts, ftilde, iter, m):

        n = 2**m
        success = False
        lnaRange = [-5, 5]
        if self.kernType == 1:
            lnaRange = [-3, 0]
        else:
            lnaRange = [-5, 5]
        # end
        # search for optimal shape parameter

        # lnaMLE = fminbnd( @ (lna) \
        #         ObjectiveFunction(self, exp(lna), xpts, ftilde), \
        #         lnaRange(1), lnaRange(2), optimset('TolX', 1e-2))
        lnaMLE = fminbnd( lambda lna: self.ObjectiveFunction(self, exp(lna), xpts, ftilde),
                x1=lnaRange[0], x2=lnaRange[1], xtol=1e-2, disp=0)

        aMLE = np.exp(lnaMLE)
        [loss, Lambda, Lambda_ring, RKHSnorm] = self.ObjectiveFunction(self, aMLE, xpts, ftilde)

        # Check error criterion
        # compute DSC
        if self.fullBayes == True:
            # full bayes
            if self.avoidCancelError:
                DSC = abs(Lambda_ring(1) / n)
            else:
                DSC = abs((Lambda(1) / n) - 1)
            # end
            # 1-alpha two sided confidence interval
            ErrBd = self.uncert * sqrt(DSC * RKHSnorm / (n - 1))
        elif self.GCV == True:
            # GCV based stopping criterion
            if self.avoidCancelError:
                DSC = abs(Lambda_ring(1) / (n + Lambda_ring(1)))
            else:
                DSC = abs(1 - (n / Lambda(1)))
            # end
            temp = Lambda
            temp[0] = n + Lambda_ring(1)
            C_Inv_trace = sum(1. / temp(temp != 0))
            ErrBd = self.uncert * sqrt(DSC * (RKHSnorm) / C_Inv_trace)

        else:
            # empirical bayes
            if self.avoidCancelError:
                DSC = abs(Lambda_ring(1) / (n + Lambda_ring(1)))
            else:
                DSC = abs(1 - (n / Lambda(1)))
            # end
            ErrBd = self.uncert * sqrt(DSC * RKHSnorm / n)
        # end

        if self.arbMean == True:  # zero mean case
            muhat = ftilde[0] / n
        else:  # non zero mean case
            muhat = ftilde[0] / Lambda[0]
        # end
        muminus = muhat - ErrBd
        muplus = muhat + ErrBd

        # store intermediate values for post analysis
        # store the debug information
        self.dscAll[iter] = sqrt(DSC)
        self.s_All[iter] = sqrt(RKHSnorm / n)
        self.muhatAll[iter] = muhat
        self.errorBdAll[iter] = ErrBd
        self.aMLEAll[iter] = aMLE
        self.lossMLEAll[iter] = loss

        if self.gaussianCheckEnable == True:
            # plots the transformed and scaled integrand values as normal plot
            # Useful to verify the assumption, integrand was an instance of a Gaussian process
            self.CheckGaussianDensity(self, ftilde, Lambda)

        # fprintf('aMLE=%1.3f, n=%d, ErrBd=%1.3f, absTol=%1.3e\n', aMLE, n, out.ErrBd, self.absTol)

        if 2 * ErrBd <= max(self.absTol, self.relTol * abs(muminus)) + max(self.absTol, self.relTol * abs(muplus)):
            if self.errorBdAll[iter] == 0:
                self.errorBdAll[iter] = np.finfo(float).eps

            # stopping criterion achieved
            success = True

        return success, muhat

    # objective function to estimate parameter theta
    # MLE : Maximum likelihood estimation
    # GCV : Generalized cross validation
    # function [loss,Lambda,Lambda_ring,RKHSnorm] =
    def ObjectiveFunction(self, a, xun, ftilde):

        n = len(ftilde)
        [Lambda, Lambda_ring] = self.kernel(xun, self.order, a, self.avoidCancelError,
                                            self.kernType, self.debugEnable)

        # compute RKHSnorm
        # temp = abs(ftilde(Lambda != 0). ** 2). / (Lambda(Lambda != 0))
        temp = abs(ftilde[Lambda != 0]**2) / (Lambda[Lambda != 0])

        # compute loss
        if self.GCV == True:
            # GCV
            # temp_gcv = abs(ftilde(Lambda != 0). / (Lambda(Lambda != 0))). ** 2
            temp_gcv = abs(ftilde[Lambda != 0] / (Lambda[Lambda != 0])) ** 2
            loss1 = 2 * log(sum(1. / Lambda[Lambda != 0]))
            loss2 = log(sum(temp_gcv[1:]))
            # ignore all zero eigenvalues
            loss = loss2 - loss1

            if self.arbMean == True:
                RKHSnorm = sum(temp_gcv[1:]) / n
            else:
                RKHSnorm = sum(temp_gcv) / n
        else:
            # default: MLE
            if self.arbMean == True:
                RKHSnorm = sum(temp[1:]) / n
                temp_1 = sum(temp[1:])
            else:
                RKHSnorm = sum(temp) / n
                temp_1 = sum(temp)

            # ignore all zero eigenvalues
            loss1 = sum(log(Lambda[Lambda != 0]))
            loss2 = n * log(temp_1)
            loss = loss1 + loss2
            # end

        if self.debugEnable:
            self.alertMsg(loss1, 'Inf')
            self.alertMsg(RKHSnorm, 'Imag')
            self.alertMsg(loss2, 'Inf')
            self.alertMsg(loss, 'Inf', 'Imag', 'Nan')
            self.alertMsg(Lambda, 'Imag')
        # end

        return loss, Lambda, Lambda_ring, RKHSnorm
    # end

    # prints debug message if the given variable is Inf, Nan or
    # complex, etc
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

            while iStart <= nargin:
                type = varargin[iStart]
                iStart = iStart + 1

                if type == 'Nan':
                    if any(np.isnan(varTocheck)):
                        print(f'{inpvarname} has NaN values')
                elif type == 'Inf':
                    if np.any(np.isinf(varTocheck)):
                        print(f'{inpvarname} has Inf values')
                elif type == 'Imag':
                    if not np.all(np.isreal(varTocheck)):
                        print(f'{inpvarname} has complex values')
                else:
                    print('unknown type check requested !')


    # computes the periodization transform for the given function values
    @staticmethod
    def doPeriodTx(fInput, ptransform):

        if ptransform == 'Baker':
            f = lambda x: fInput(1 - 2 * abs(x - 1 / 2))  # Baker's transform
        elif ptransform == 'C0':
            f = lambda x: fInput(3 * x ** 2 - 2 * x ** 3) * np.prod(6 * x * (1 - x), 2)  # C^0 transform
        elif ptransform == 'C1':
            # C^1 transform
            f = lambda x: fInput(x ** 3 * (10 - 15 * x + 6 * x ** 2)) * np.prod(30 * x ** 2 * (1 - x) ** 2, 2)
        elif ptransform == 'C1np.sin':
            # Sidi C^1 transform
            f = lambda x: fInput(x - np.sin(2 * np.pi * x) / (2 * np.pi)) * np.prod(2 * np.sin(np.pi * x) ** 2, 2)
        elif ptransform == 'C2np.sin':
            # Sidi C^2 transform
            psi3 = lambda t: (8 - 9 * np.cos(np.pi * t) + np.cos(3 * np.pi * t)) / 16
            psi3_1 = lambda t: (9 * np.sin(np.pi * t) * np.pi - np.sin(3 * np.pi * t) * 3 * np.pi) / 16
            f = lambda x: fInput(psi3(x)) * np.prod(psi3_1(x), 2)
        elif ptransform == 'C3np.sin':
            # Sidi C^3 transform
            psi4 = lambda t: (12 * np.pi * t - 8 * np.sin(2 * np.pi * t) + np.sin(4 * np.pi * t)) / (12 * np.pi)
            psi4_1 = lambda t: (12 * np.pi - 8 * np.cos(2 * np.pi * t) * 2 * np.pi + np.sin(4 * np.pi * t) * 4 * np.pi) / (12 * np.pi)
            f = lambda x: fInput(psi4(x)) * np.prod(psi4_1(x), 2)
        elif ptransform == 'none':
            # do nothing
            f = lambda x: fInput(x)
        else:
            print(f'Error: Periodization transform {ptransform} not implemented')

        return f


    # Computes modified kernel Km1 = K - 1
    # Useful to avoid cancellation error in the computation of (1 - n/\lambda_1)
    @staticmethod
    def kernel_t(aconst, Bern):
        theta = aconst
        d = np.size(Bern, 2)

        Kjm1 = theta * Bern[:, 1]  # Kernel at j-dim minus One
        Kj = 1 + Kjm1  # Kernel at j-dim

        for j in range(2, d):
            Kjm1_prev = Kjm1
            Kj_prev = Kj  # save the Kernel at the prev dim

            Kjm1 = theta * Bern[:, j] * Kj_prev + Kjm1_prev
            Kj = 1 + Kjm1

        Km1 = Kjm1
        K = Kj
        return [Km1, K]


    # Shift invariant kernel
    # C1 : first row of the covariance matrix
    # Lambda : eigen values of the covariance matrix
    # Lambda_ring = fft(C1 - 1)
    @staticmethod
    def kernel(xun, order, a, avoidCancelError, kernType, debug_enable):

        if kernType == 1:
            b_order = order * 2  # Bernoulli polynomial order as per the equation
            constMult = -(-1) ** (b_order / 2) * ((2 * np.pi) ** b_order) / np.factorial(b_order)
            # constMult = -(-1)**(b_order/2)
            if b_order == 2:
                bernPoly = lambda x: (-x* (1 - x) + 1 / 6)
            elif b_order == 4:
                bernPoly = lambda x: (((x* (1 - x))** 2) - 1 / 30)
            else:
                print('Error: Bernoulli order not implemented !')

            kernelFunc = lambda x: bernPoly(x)
        else:
            b = order
            kernelFunc = lambda x: 2 * b * ((np.cos(2 * np.pi * x) - b))/ (1 + b ^ 2 - 2 * b * np.cos(2 * np.pi * x))
            constMult = 1

        if avoidCancelError:
            # Computes C1m1 = C1 - 1
            # C1_new = 1 + C1m1 indirectly computed in the process
            (C1m1, C1_alt) = CubBayesLattice_g.kernel_t(a * constMult, kernelFunc(xun))
            # eigenvalues must be real : Symmetric pos definite Kernel
            Lambda_ring = np.real(np.fft(C1m1))

            Lambda = Lambda_ring
            Lambda[0] = Lambda_ring(1) + len(Lambda_ring)
            # C1 = prod(1 + (a)*constMult*bernPoly(xun),2)  # direct computation
            # Lambda = real(fft(C1))

            if debug_enable == True:
                # eigenvalues must be real : Symmetric pos definite Kernel
                Lambda_direct = np.real(np.fft(C1_alt))  # Note: fft output unnormalized
                if sum(abs(Lambda_direct - Lambda)) > 1:
                    print('Possible error: check Lambda_ring computation')

        else:
            # direct approach to compute first row of the kernel Gram matrix
            C1 = np.prod(1 + (a) * constMult * kernelFunc(xun), 2)
            # matlab's builtin fft is much faster and accurate
            # eigenvalues must be real : Symmetric pos definite Kernel
            Lambda = np.real(np.fft(C1))
            Lambda_ring = 0

        return Lambda, Lambda_ring

    # just returns the generator for rank-1 Lattice point generation
    def get_lattice_gen_vec(d):
        z = [1, 433461, 315689, 441789, 501101, 146355, 88411, 215837, 273599,
             151719, 258185, 357967, 96407, 203741, 211709, 135719, 100779,
             85729, 14597, 94813, 422013, 484367]  # generator
        z = z[0:d-1]
        return z

    # generates rank-1 Lattice points in Vander Corput sequence order
    def simple_lattice_gen(n, d, shift, firstBatch):
        z = CubBayesLattice_g.get_lattice_gen_vec(d)

        nmax = n
        nmin = 1 + n / 2
        if firstBatch == True:
            nmin = 1

        nelem = nmax - nmin + 1

        if firstBatch == True:
            brIndices = CubBayesLattice_g.vdc(nelem).transpose()

            # xlat_ = np.mod(bsxfun( @ times, (0:1 / n:1-1 / n)',z),1) # unshifted in direct order
            xlat_ = np.mod( (np.arange(0, 1-1/n, 1/n).transpose() * z), 1)
        else:
            brIndices = CubBayesLattice_g.vdc(nelem).transpose() + 1/(2*(nmin-1))
            # xlat_ = mod(bsxfun( @ times, (1 / n:2 / n:1-1 / n)',z),1) # unshifted in direct order
            xlat_ = np.mod((np.arange(1/n, 1-1/n, 2/n).transpose() * z), 1)

        # xlat = mod(bsxfun( @ times, brIndices',z),1)  # unshifted
        xlat = np.mod((brIndices.transpose() * z), 1)
        # xlat = mod(bsxfun( @ plus, xlat, shift), 1)  # shifted in VDC order
        xlat = np.mod((xlat + shift), 1)

        return [xlat, xlat_]

    # van der Corput sequence in base 2
    def vdc(n):
        if n > 1:
            k = np.log2(n)  # We compute the VDC seq part by part of power 2 size
            q = np.zeros(2**k, 1)
            for l in range(0, k-1):
                nl = 2 ** l
                kk = 2 ** (k - l - 1)
                ptind = np.matlib.repmat(np.vstack(np.full((nl, 1), False), np.full((nl, 1), True)), kk, 1)
                q[ptind] = q[ptind] + 1 / 2 ** (l + 1)
        else:
            q = 0

        return q

    # fft with decimation in time i.e., input is already in 'bitrevorder'
    def fft_DIT(y, nmmin):
        for l in range(0, nmmin - 1):
            nl = 2 ** l
            nmminlm1 = 2 ** (nmmin - l - 1)
            # ptind=repmat([true(nl,1); false(nl,1)], nmminlm1,1)
            ptind = np.matlib.repmat(np.vstack(np.full((nl, 1), True), np.full((nl, 1), False)), nmminlm1, 1)
            # coef=exp(-2*pi()*sqrt(-1)*(0:nl-1)'/(2*nl))
            coef = exp(-2 * np.pi * 1j * np.arange(0, nl-1).transpose()/(2*nl))
            coefv = np.matlib.repmat(coef, nmminlm1, 1)
            evenval = y[ptind]
            oddval = y[~ptind]
            y[ptind] = (evenval + coefv * oddval)
            y[~ptind] = (evenval - coefv * oddval)

        return y


    # using FFT butefly plot technique merges two halves of fft
    def merge_fft(ftildeNew, ftildeNextNew, mnext):
        ftildeNew = np.vstack(ftildeNew, ftildeNextNew)
        nl = 2**mnext
        # ptind=[true(nl,1); false(nl,1)]
        ptind = np.ndarray(shape=(2*nl,1), buffer=np.array( [True]*nl + [False]*nl), dtype=bool)
        # coef = exp(-2*1j*(0:nl-1)'/(2*nl))
        coef = exp(-2 * 1j * np.ndarray(shape=(nl, 1), buffer=np.arange(0, nl-1), dtype=int)/(2*nl))
        coefv = np.matlib.repmat(coef, 1, 1)
        evenval = ftildeNew[ptind]
        oddval = ftildeNew[~ptind]
        ftildeNew[ptind] = (evenval+coefv*oddval)
        ftildeNew[~ptind] = (evenval-coefv*oddval)
        return ftildeNew


    # inserts newly generated points with the old set by interleaving them
    # xun - unshifted points
    def merge_pts(xun, xunnew, x, xnew, n, d):
        temp = np.zeros((n,d))
        temp[1:2:n-1,:] = xun
        temp[2:2:n,:] = xunnew
        xun = temp
        temp[1:2:n-1,:] = x
        temp[2:2:n,:] = xnew
        x = temp
        return xun, x