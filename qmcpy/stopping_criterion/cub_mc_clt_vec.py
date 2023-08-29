from ._stopping_criterion import StoppingCriterion
from .cub_mc_clt import CubMCCLT
from ..accumulate_data import MeanVarData
from ..discrete_distribution import IIDStdUniform
from ..discrete_distribution._discrete_distribution import IID
from ..true_measure import Gaussian, BrownianMotion, Uniform
from ..integrand import Keister, AsianOption, CustomFun
from ..util import MaxSamplesWarning
from numpy import *
from scipy.stats import norm
from time import time
import warnings

class CubMCCLTVec(StoppingCriterion):
    def __init__(self, integrand, abs_tol=1e-2, rel_tol=0., n_init=1024., n_max=1e10,
        inflate=1.2, alpha=0.01, control_variates=[], control_variate_means=[],
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol)):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (ndarray): absolute error tolerance
            rel_tol (ndarray): relative error tolerance
            n_max (int): maximum number of samples
            control_variates (list): list of integrand objects to be used as control variates. 
                Control variates are currently only compatible with single level problems. 
                The same discrete distribution instance must be used for the integrand and each of the control variates. 
            control_variate_means (list): list of means for each control variate
        """

        self.parameters = ['abs_tol','rel_tol','n_init','n_max','inflate','alpha']
        # Set Attributes
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.n_init = n_init
        self.n_max = n_max
        self.alpha = alpha
        self.inflate = inflate
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.cv = control_variates
        self.cv_mu = control_variate_means
        # Verify Compliant Construction
        allowed_levels = ['single','fixed-multi']
        allowed_distribs = [IID]
        allow_vectorized_integrals = True
        super(CubMCCLTVec,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

    def integrate(self):
        """
            x = discrete_distribution.gen_samples(n)
            print("x = " + str(x))
            y = fn(x[:,d])
            print("y = " + str(y))
            mean = y.mean()
            print("mean = " + str(mean))
            variance = y.var(ddof=1)
            print("variance = " + str(variance))
            margin_error = sqrt(inflation) * norm.ppf(level) * (sqrt(variance/n))
            print("margin of error = " + str(margin_error))
            conf_int_lbound = mean - margin_error
            print("lower bound = " + str(conf_int_lbound))
            conf_int_hbound = mean + margin_error
            print("upper bound = " + str(conf_int_hbound))
            return conf_int_lbound,conf_int_hbound
        """
         # Construct AccumulateData Object to House Integration data
        self.data = MeanVarData(self, self.integrand, self.true_measure, self.discrete_distrib, 
            self.n_init, self.cv, self.cv_mu)  # house integration data
        t_start = time()
        # Pilot Sample
        self.data.update_data()
        # use cost of function values to decide how to allocate
        temp_a = self.data.t_eval ** 0.5
        temp_b = (temp_a * self.data.sighat).sum()
        # samples for computation of the mean
        # n_mu_temp := n such that confidence intervals width and conficence will be satisfied
        tol_up = max(self.abs_tol, abs(self.data.solution) * self.rel_tol)
        z_star = -norm.ppf(self.alpha / 2.)
        n_mu_temp = ceil(temp_b * (self.data.sighat / temp_a) * (z_star * self.inflate / tol_up)**2)
        # n_mu := n_mu_temp adjusted for previous n
        self.data.n_mu = maximum(self.data.n, n_mu_temp)
        self.data.n += self.data.n_mu.astype(int)
        if self.data.n_total + self.data.n.sum() > self.n_max:
            # cannot generate this many new samples
            warning_s = """
            Alread generated %d samples.
            Trying to generate %d new samples, which would exceed n_max = %d.
            The number of new samples will be decrease proportionally for each integrand.
            Note that error tolerances may no longer be satisfied.""" \
            % (int(self.data.n_total), int(self.data.n.sum()), int(self.n_max))
            warnings.warn(warning_s, MaxSamplesWarning)
            # decrease n proportionally for each integrand
            n_decease = self.data.n_total + self.data.n.sum() - self.n_max
            dec_prop = n_decease / self.data.n.sum()
            self.data.n = floor(self.data.n - self.data.n * dec_prop)
        # Final Sample
        self.data.update_data()
        # CLT Confidence Interval
        print("mean = " + str(self.data.muhat))
        print("std = " + str(self.data.sighat))
        print("samples = " + str(self.data.n_total))
        margin_error = sqrt(self.inflate) * z_star * (self.data.sighat / (sqrt(self.data.n_total)))
        print("margin of error = " + str(margin_error))
        conf_int_lbound = self.data.muhat - margin_error
        print("lower bound = " + str(conf_int_lbound))
        conf_int_hbound = self.data.muhat + margin_error
        print("upper bound = " + str(conf_int_hbound))
        self.data.time_integrate = time() - t_start
        return conf_int_lbound,conf_int_hbound



    