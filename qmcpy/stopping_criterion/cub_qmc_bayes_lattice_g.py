from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import LDTransformBayesData
from ..discrete_distribution import Lattice
from ..true_measure import Gaussian
from ..integrand import Keister
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning
from numpy import sqrt, log2, hstack, tile, exp, pi, arange, log
import numpy as np
from time import time
from scipy.optimize import fminbound as fminbnd
import warnings


class CubBayesLatticeG(StoppingCriterion):
    """
    Stopping criterion for Bayesian Cubature using rank-1 Lattice sequence with guaranteed
    accuracy over a d-dimensional region to integrate within a specified generalized error
    tolerance with guarantees under Bayesian assumptions.
    
    >>> k = Keister(Gaussian(Lattice(2,seed=7),covariance=1./2))
    >>> sc = CubBayesLatticeG(k,abs_tol=.05)
    >>> solution,data = sc.integrate()
    >>> solution
    1.808134131740979
    >>> data
    Solution: 1.8081         
    Keister (Integrand Object)
    Lattice (DiscreteDistribution Object)
        dimension       2^(1)
        randomize       1
        seed            7
        backend         gail
        mimics          StdUniform
    Gaussian (TrueMeasure Object)
        mean            0
        covariance      2^(-1)
    CubQMCLatticeG (StoppingCriterion Object)
        abs_tol         0.050
        rel_tol         0
        n_init          2^(10)
        n_max           2^(35)
    LDTransformData (AccumulateData Object)
        n_total         2^(10)
        solution        1.808
        r_lag           2^(2)
        time_integrate  ...

    Adapted from 
        https://github.com/GailGithub/GAIL_Dev/blob/master/Algorithms/IntegrationExpectation/CubBayesLatticeG.m

    Reference
        [1] Sou-Cheng T. Choi, Yuhan Ding, Fred J. Hickernell, Lan Jiang, Lluis Antoni Jimenez Rugama,
        Da Li, Jagadeeswaran Rathinavel, Xin Tong, Kan Zhang, Yizhi Zhang, and Xuan Zhou, 
        GAIL: Guaranteed Automatic Integration Library (Version 2.3) [MATLAB Software], 2019. 
        Available from http://gailgithub.github.io/GAIL_Dev/

    Guarantee
 
        
        This algorithm attempts to calculate the integral of function f over the
        hyperbox [0,1]^d to a prescribed error tolerance tolfun:= max(abstol,reltol*| I |)
        with guaranteed confidence level, e.g., 99% when alpha=0.5%. If the
        algorithm terminates without showing any warning messages and provides
        an answer Q, then the following inequality would be satisfied:
     
                Pr(| Q - I | <= tolfun) = 99%

        This Bayesian cubature algorithm guarantees for integrands that are considered to be an instance of a
        gaussian process that fall in the middle of samples space spanned.
        Where The sample space is spanned by the covariance kernel parametrized by the scale and shape parameter
        inferred from the sampled values of the integrand.
        For more details on how the covariance kernels are defined and the parameters are obtained, please
        refer to the references below.
    """
    
    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0,
                 n_init=2**10, n_max=2**22, alpha=0.01):
        # Set Attributes
        self.absTol = abs_tol
        self.relTol = rel_tol
        m_min = log2(n_init)
        m_max = log2(n_max)
        self.n_min = n_init  # number of samples to start with = 2^mmin
        self.n_max = n_max  # max number of samples allowed = 2^mmax
        self.alpha = alpha  # p-value, default 0.1%.
        self.order = 2  # Bernoulli kernel's order. If zero, choose order automatically

        self.f = lambda x: x**2  # function to integrate
        self.dim = 1  # dimension of the integrand
        self.useGradient = False  # If true uses gradient descent in parameter search
        self.oneTheta = True  # If true use common shape parameter for all dimensions
        # else allow shape parameter vary across dimensions
        self.ptransform = 'C1sin'  # periodization transform
        self.stopAtTol = True  # automatic mode: stop after meeting the error tolerance
        self.arbMean = True  # by default use zero mean algorithm
        self.stopCriterion = 'MLE'  # Available options {'MLE', 'GCV', 'full'}

        # private properties
        self.fullBayes = False  # Full Bayes - assumes m and s^2 as hyperparameters
        self.GCV = False  # Generalized cross validation
        self.vdc_order = False  # use Lattice points generated in vdc order
        self.kernType = 1  # Type-1: Bernoulli polynomial based algebraic convergence
        # Type-2: Truncated series
        self.avoidCancelError = True  # avoid cancellation error in stopping criterion
        self.uncert = 0  # quantile value for the error bound
        self.debugEnable = False  # enable debug prints

        # variables to save debug info in each iteration
        self.errorBdAll = []
        self.muhatAll = []
        self.aMLEAll = []
        self.lossMLEAll = []
        self.timeAll = []
        self.dscAll = []
        self.s_All = []

        '''
        ff = []  # integrand after the periodization transform
        mvec = []  #  n = 2^m
        gen_vec = []  #  generator for the Lattice points
                
        % only for developers use
        fName = 'None'  # name of the integrand
        figSavePath = ''  # path where to save he figures
        visiblePlot = false  # make plots visible
        gaussianCheckEnable = false  # enable plot to check Gaussian pdf
        '''

        # Verify Compliant Construction
        distribution = integrand.measure.distribution
        allowed_levels = 'single'
        allowed_distribs = ["Lattice"]
        super(CubBayesLatticeG, self).__init__(distribution, allowed_levels, allowed_distribs)

        if not distribution.randomize:
            raise ParameterError("CubLattice_g requires distribution to have randomize=True")
        if distribution.backend != 'gail':
            raise ParameterError("CubLattice_g requires distribution to have 'GAIL' backend")
        # Construct AccumulateData Object to House Integration data
        self.data = LDTransformBayesData(self, integrand, self.fft_update, m_min, m_max, fudge, check_cone)

    def integrate_ref(self):
        """ See abstract method. """
        t_start = time()
        while True:
            self.data.update_data()
            # Check the end of the algorithm
            errest = self.data.fudge(self.data.m) * self.data.stilde
            # Compute optimal estimator
            ub = max(self.absTol, self.relTol * abs(self.data.solution + errest))
            lb = max(self.absTol, self.relTol * abs(self.data.solution - errest))
            self.data.solution = self.data.solution - errest * (ub - lb) / (ub + lb)
            if 4. * errest ** 2 / (ub + lb) ** 2 <= 1.:
                # stopping criterion met
                break
            elif self.data.m == self.data.m_max:
                # doubling samples would go over n_max
                warning_s = """
                    Already generated %d samples.
                    Trying to generate %d new samples would exceed n_max = %d.
                    No more samples will be generated.
                    Note that error tolerances may no longer be satisfied""" \
                            % (int(2. ** self.data.m), int(self.data.m), int(2. ** self.data.m_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                # double sample size
                self.data.m += 1.
        self.data.time_integrate = time() - t_start
        return self.data.solution, self.data

    ## Efficient FFT computation algorithm, avoids recomputing the full fft
    def iter_fft(self, iter, shift, xun, xpts, ftildePrev):
        m = self.mvec(iter)
        n = 2**m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FFT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 1:
            # In the first iteration compute full FFT
            # xun_ = mod(bsxfun( @ times, (0 : 1/n : 1 - 1/n)',obj.gen_vec),1)
            xun_ = np.mod( np.arange(0, 1 - 1/n, 1/n).transpose() * self.gen_vec,1)
            # xpts_ = mod(bsxfun( @ plus, xun_, shift), 1)  # shifted
            xpts_ = np.mod(xun_ + shift, 1)  # shifted

            # Compute initial FFT
            ftilde_ = np.fft(self.gpuArray_(self.ff(xpts_)))  # evaluate integrand's fft
        else:
            # xunnew = mod(bsxfun( @ times, (1 / n:2 / n:1-1 / n)',obj.gen_vec),1)
            xunnew = np.mod((  np.arange(1/n, 1-1/n, 2 / n).transpose() * self.gen_vec),1)

            # xnew = mod(bsxfun( @ plus, xunnew, shift), 1)
            xnew = np.mod((xunnew + shift), 1)

            xun_, xpts_ = self.merge_pts(xun, xunnew, xpts, xnew, n, obj.dim)
            mnext = m - 1

            # Compute FFT on next set of new points
            ftildeNextNew = np.fft(self.gpuArray_(self.ff(xnew)))
            if self.debugEnable:
                CubBayesLatticeG.alertMsg(ftildeNextNew, 'Nan', 'Inf')
            # end

            # combine the previous batch and new batch to get FFT on all points
            ftilde_ = self.merge_fft(ftildePrev, ftildeNextNew, mnext)
            # end

            # function [ftilde_,xun_,xpts_] =
        return ftilde_, xun_, xpts_

    # end

    # Lattice points are ordered in van der Corput sequence, so we cannot use
    # Matlab's built-in fft routine. We use a custom one instead.
    def iter_fft_vdc(obj, iter, shift, xun, xpts, ftildePrev):
        m = obj.mvec(iter)
        n = 2**m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FFT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 1:
            # in the first iteration compute the full FFT
            xpts_, xun_ = obj.simple_lattice_gen(n, obj.dim, shift, True)

            # Compute initial FFT
            ftilde_ = obj.fft_DIT(obj.gpuArray_(obj.ff(xpts_)), m)  # evaluate integrand's fft
        else:
            xnew, xunnew = obj.simple_lattice_gen(n, obj.dim, shift, False)
            mnext = m - 1

            # Compute FFT on next set of new points
            ftildeNextNew = obj.fft_DIT(obj.gpuArray_(obj.ff(xnew)), mnext)
            if obj.debugEnable:
                CubBayesLatticeG.alertMsg(ftildeNextNew, 'Nan', 'Inf')
            # end

            xpts_ = np.vstack(xpts, xnew)
            temp = np.zeros(n, obj.dim)
            temp[1: 2:n - 1,:] = xun
            temp[2: 2:n,:] = xunnew
            xun_ = temp
            # combine the previous batch and new batch to get FFT on all points
            ftilde_ = obj.merge_fft(ftildePrev, ftildeNextNew, mnext)
        # end
        if obj.debugEnable:
            CubBayesLatticeG.alertMsg(ftilde_, 'Inf', 'Nan')
        # end

        # [ftilde_,xun_,xpts_]
        return ftilde_, xun_, xpts_

    # end

    ## Efficient FFT computation algorithm, avoids recomputing the full fft
    def iter_fft(self, iter, shift, xun, xpts, ftildePrev):
        m = self.mvec(iter)
        n = 2**m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FFT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 1:
            # In the first iteration compute full FFT
            #xun_ = mod(bsxfun( @ times, (0:1 / n:1-1 / n)',self.gen_vec),1)
            xun_ = np.mod(( np.arange(0, 1-1/n, 1/n).transpose() * self.gen_vec),1)

            # xpts_ = np.mod(bsxfun( @ plus, xun_, shift), 1)  # shifted
            xpts_ = np.mod(( xun_ + shift), 1)  # shifted

            # Compute initial FFT
            ftilde_ = np.fft(self.gpuArray_(self.ff(xpts_)))  # evaluate integrand's fft
        else:
            # xunnew = np.mod(bsxfun( @ times, (1/n : 2/n : 1-1/n)',self.gen_vec),1)
            xunnew = np.mod( np.arange(1/n, 1 - 1/n, 2/n)* self.gen_vec, 1)

            # xnew = np.mod(bsxfun( @ plus, xunnew, shift), 1)
            xnew = np.mod((xunnew + shift), 1)

            [xun_, xpts_] = self.merge_pts(xun, xunnew, xpts, xnew, n, self.dim)
            mnext = m - 1

            # Compute FFT on next set of new points
            ftildeNextNew = np.fft(self.gpuArray_(self.ff(xnew)))
            if self.debugEnable:
                CubBayesLatticeG.alertMsg(ftildeNextNew, 'Nan', 'Inf')
            # end

            # combine the previous batch and new batch to get FFT on all points
            ftilde_ = self.merge_fft(ftildePrev, ftildeNextNew, mnext)
            # end

        # function [ftilde_,xun_,xpts_] =
        return ftilde_, xun_, xpts_

    # end

    # Lattice points are ordered in van der Corput sequence, so we cannot use
    # Matlab's built-in fft routine. We use a custom one instead.
    def iter_fft_vdc(self, iter, shift, xun, xpts, ftildePrev):
        m = self.mvec(iter)
        n = 2**m

        # In every iteration except the first one, "n" number_of_points is doubled,
        # but FFT is only computed for the newly added points.
        # Previously computed FFT is reused.
        if iter == 1:
            # in the first iteration compute the full FFT
            [xpts_, xun_] = self.simple_lattice_gen(n, self.dim, shift, True)

            # Compute initial FFT
            ftilde_ = self.fft_DIT(self.gpuArray_(self.ff(xpts_)), m)  # evaluate integrand's fft
        else:
            [xnew, xunnew] = self.simple_lattice_gen(n, self.dim, shift, False)
            mnext = m - 1

            # Compute FFT on next set of new points
            ftildeNextNew = self.fft_DIT(self.gpuArray_(self.ff(xnew)), mnext)
            if self.debugEnable:
                CubBayesLatticeG.alertMsg(ftildeNextNew, 'Nan', 'Inf')
            # end

            xpts_ = np.vstack(xpts, xnew)
            temp = np.zeros(n, self.dim)
            temp[1: 2:n - 1,:] = xun
            temp[2: 2:n,:] = xunnew
            xun_ = temp
            # combine the previous batch and new batch to get FFT on all points
            ftilde_ = self.merge_fft(ftildePrev, ftildeNextNew, mnext)
        # end
        if self.debugEnable:
            CubBayesLatticeG.alertMsg(ftilde_, 'Inf', 'Nan')
        # end

        # [ftilde_,xun_,xpts_]
        return ftilde_, xun_, xpts_

    # end



    # computes the integral
    def integrate(self):

        tstart = time() # tstart = tic  # start the clock
        numM = len(self.mvec)

        # pick a random value to apply as shift
        shift = np.random.rand(1, self.dim)

        xun_ = []
        xpts_ = []
        ftilde_ = []  # temporary storage between iterations
        ## Iteratively find the number of points required for the cubature to meet
        # the error threshold
        for iter in range(1, numM):
            tstart_iter = time()
            m = self.mvec[iter]
            n = 2**m

            # Update function values
            if self.vdc_order:
                ftilde_, xun_, xpts_ = self.iter_fft_vdc(iter, shift, xun_, xpts_, ftilde_)
            else:
                ftilde_, xun_, xpts_ = self.iter_fft(iter, shift, xun_, xpts_, ftilde_)
            # end
            stop_flag, muhat, order_ = self.stopping_criterion(xun_, ftilde_, iter, m)

            self.timeAll[iter] = time() - tstart_iter  # store per iteration time

            # if stopAtTol true, exit the loop
            # else, run for for all 'n' values.
            # Used to compute error values for 'n' vs error plotting
            if self.stopAtTol == True and stop_flag == True:
                break
            # end

        # end
        out.n = n
        out.time = tstart - time()
        out.ErrBd = self.errorBdAll[-1]

        optParams.ErrBdAll = self.errorBdAll
        optParams.muhatAll = self.muhatAll
        optParams.mvec = self.mvec
        optParams.aMLEAll = self.aMLEAll
        optParams.timeAll = self.timeAll
        optParams.s_All = self.s_All
        optParams.dscAll = self.dscAll
        optParams.absTol = self.absTol
        optParams.relTol = self.relTol
        optParams.shift = shift
        optParams.stopAtTol = self.stopAtTol
        optParams.r = order_
        out.optParams = optParams
        if stop_flag == True:
            out.exitflag = 1
        else:
            out.exitflag = 2  # error tolerance may not be met
        # end

        if stop_flag == False:
            warning('GAIL:cubBayesLattice_g:maxreached', ...
            ['In order to achieve the guaranteed accuracy, ', ...
            sprintf('used maximum allowed sample size %d. \n', n)] )
        # end

        # convert from gpu memory to local
        muhat = self.gather_(muhat)
        out = self.gather_(out)

        # function [muhat,out] =
        return [muhat, out]
    # end






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
        
        # search for optimal shape parameter
        # lnaMLE = fminbnd( @ (lna) \
        #         ObjectiveFunction(self, exp(lna), xpts, ftilde), \
        #         lnaRange(1), lnaRange(2), optimset('TolX', 1e-2))
        lnaMLE = fminbnd( lambda lna: self.ObjectiveFunction(exp(lna), xpts, ftilde),
                x1=lnaRange[0], x2=lnaRange[1], xtol=1e-2, disp=0)

        aMLE = exp(lnaMLE)
        [loss, Lambda, Lambda_ring, RKHSnorm] = self.ObjectiveFunction(aMLE, xpts, ftilde)

        # Check error criterion
        # compute DSC
        if self.fullBayes == True:
            # full Bayes
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
            # empirical Bayes
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


        if 2 * ErrBd <= max(self.absTol, self.relTol * abs(muminus)) + max(self.absTol, self.relTol * abs(muplus)):
            if self.errorBdAll[iter] == 0:
                self.errorBdAll[iter] = np.finfo(float).eps

            # stopping criterion achieved
            success = True

        return success, muhat

    # objective function to estimate parameter theta
    # MLE : Maximum likelihood estimation
    # GCV : Generalized cross validation
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

        # function [loss,Lambda,Lambda_ring,RKHSnorm] =
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


    '''
    Shift invariant kernel
    C1 : first row of the covariance matrix
    Lambda : eigen values of the covariance matrix
    Lambda_ring = fft(C1 - 1)
    '''
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
            kernelFunc = lambda x: 2 * b * ((np.cos(2 * np.pi * x) - b))/ (1 + b**2 - 2 * b * np.cos(2 * np.pi * x))
            constMult = 1

        if avoidCancelError:
            # Computes C1m1 = C1 - 1
            # C1_new = 1 + C1m1 indirectly computed in the process
            (C1m1, C1_alt) = CubBayesLatticeG.kernel_t(a * constMult, kernelFunc(xun))
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
    @staticmethod
    def get_lattice_gen_vec(d):
        z = [1, 433461, 315689, 441789, 501101, 146355, 88411, 215837, 273599,
             151719, 258185, 357967, 96407, 203741, 211709, 135719, 100779,
             85729, 14597, 94813, 422013, 484367]  # generator
        z = z[0:d-1]
        return z

    # generates rank-1 Lattice points in Vander Corput sequence order
    @staticmethod
    def simple_lattice_gen(n, d, shift, firstBatch):
        z = CubBayesLatticeG.get_lattice_gen_vec(d)

        nmax = n
        nmin = 1 + n / 2
        if firstBatch == True:
            nmin = 1

        nelem = nmax - nmin + 1

        if firstBatch == True:
            brIndices = CubBayesLatticeG.vdc(nelem).transpose()

            # xlat_ = np.mod(bsxfun( @ times, (0:1 / n:1-1 / n)',z),1) # unshifted in direct order
            xlat_ = np.mod( (np.arange(0, 1-1/n, 1/n).transpose() * z), 1)
        else:
            brIndices = CubBayesLatticeG.vdc(nelem).transpose() + 1/(2*(nmin-1))
            # xlat_ = mod(bsxfun( @ times, (1 / n:2 / n:1-1 / n)',z),1) # unshifted in direct order
            xlat_ = np.mod((np.arange(1/n, 1-1/n, 2/n).transpose() * z), 1)

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
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def merge_pts(xun, xunnew, x, xnew, n, d):
        temp = np.zeros((n,d))
        temp[1:2:n-1,:] = xun
        temp[2:2:n,:] = xunnew
        xun = temp
        temp[1:2:n-1,:] = x
        temp[2:2:n,:] = xnew
        x = temp
        return xun, x