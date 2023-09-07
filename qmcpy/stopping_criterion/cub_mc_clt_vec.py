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
        inflate=1.2, alpha=0.01,error_fun = lambda sv,abs_tol,rel_tol: maximum(abs_tol,abs(sv)*rel_tol)):
        """
        Args:
            integrand (Integrand): an instance of Integrand
            inflate (float): inflation factor when estimating variance
            alpha (float): significance level for confidence interval
            abs_tol (ndarray): absolute error tolerance
            rel_tol (ndarray): relative error tolerance
            n_max (int): maximum number of samples
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
        # Verify Compliant Construction
        allowed_levels = ['single']
        allowed_distribs = [IID]
        allow_vectorized_integrals = True
        super(CubMCCLTVec,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)

    def integrate(self):
        
        self.data = MeanVarData(self, self.integrand, self.true_measure, self.discrete_distrib, 
            self.n_init, [], [])  
        t_start = time()
        z_star = -norm.ppf(self.alpha/2)
        self.data.error_bound = self.abs_tol
        while(self.data.error_bound >= self.abs_tol and self.n_init <= self.n_max):
            self.data.update_data()
            print("mean = " + str(self.data.muhat))
            print("std = " + str(self.data.sighat))
            self.data.error_bound = sqrt(self.inflate) * z_star * (self.data.sighat / (sqrt(self.data.n)))
            print("margin of error = " + str(self.data.error_bound))
            self.data.confid_int = array([self.data.muhat - self.data.error_bound,self.data.muhat + self.data.error_bound])
            print(" confidence interval = " + str(self.data.confid_int))
            print("confidence half width = " + str(self.data.error_bound))
            if(self.data.n < self.n_max and (self.data.n * 2) > self.n_max):
                self.data.n = tile(self.n_max,self.data.levels)
            else:
                self.data.n = tile(self.data.n * 2, self.data.levels)
        self.data.time_integrate = time() - t_start
        return self.data.solution, self.data
        
        


    