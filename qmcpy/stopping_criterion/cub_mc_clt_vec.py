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

class CubMCCLTVec(CubMCCLT):
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

        super(CubMCCLTVec,self).__init__(integrand, abs_tol=1e-2, rel_tol=0., n_init=1024., n_max=1e10,
        inflate=1.2, alpha=0.01, control_variates=[], control_variate_means=[],
        error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol))

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
        super(CubMCCLTVec,self).integrate()
        print("mean = " + str(self.data.muhat))
        print("std = " + str(self.data.sighat))
        print("samples = " + str(self.data.n_total))
        margin_error = sqrt(self.inflate) * -norm.ppf(self.alpha / 2) * (self.data.sighat / (sqrt(self.data.n_total)))
        print("margin of error = " + str(margin_error))
        conf_int_lbound = self.data.muhat - margin_error
        print("lower bound = " + str(conf_int_lbound))
        conf_int_hbound = self.data.muhat + margin_error
        print("upper bound = " + str(conf_int_hbound))
        return conf_int_lbound,conf_int_hbound



    