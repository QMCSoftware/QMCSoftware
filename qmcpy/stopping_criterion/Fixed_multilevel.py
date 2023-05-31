#!/usr/bin/env python
# coding: utf-8

# In[5]:


from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import MLQMCData
from ..discrete_distribution import DigitalNetB2,Lattice,Halton
from ..discrete_distribution._discrete_distribution import LD
from ..true_measure import Gaussian
from ..integrand import MLCallOptions
from ..util import MaxSamplesWarning, ParameterError, MaxLevelsWarning
from numpy import *
from scipy.stats import norm
from time import time
import warnings


class CubQMCFML(StoppingCriterion):
    
    def __init__(self, integrand, abs_tol=.05, alpha=.01, rmse_tol=None, n_init=256., n_max=1e10, 
        replications=32., levels=3):
        """
        Args:
            integrand (Integrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance
            alpha (float): uncertaintly level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rmse_tol (float): root mean squared error
                If supplied (not None), then absolute tolerance and alpha are ignored
                in favor of the rmse tolerance
            n_max (int): maximum number of samples
            replications (int): number of replications on each level
            levels (int): number of levels of refinement >= 2

        """
        self.parameters = ['rmse_tol','n_init','n_max','replications']
        # initialization
        if rmse_tol:
            self.rmse_tol = float(rmse_tol)
        else: # use absolute tolerance
            self.rmse_tol =  float(abs_tol) / norm.ppf(1-alpha/2)
        self.n_init = float(n_init)
        self.n_max = float(n_max)
        self.replications = float(replications)
        self.levels = levels
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        # Verify Compliant Construction
        allowed_levels = ['fixed-multi']
        allowed_distribs = [LD]
        allow_vectorized_integrals = False
        super(CubQMCFML,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals)
        
    def integrate(self):
        # Construct AccumulateData Object to House Integration Data
        self.data = MLQMCData(self, self.integrand, self.true_measure, self.discrete_distrib,
            self.levels, self.levels, self.n_init, self.replications)
        t_start = time()
        while True:
            self.data.update_data()
            if self.data.var_level.sum() > (self.rmse_tol**2/2.):
                # double N_l on level with largest V_l/(2^l*N_l)
                efficient_level = argmax(self.data.var_cost_ratio_level)
                self.data.eval_level[efficient_level] = True
            elif self.data.bias_estimate > (self.rmse_tol/sqrt(2.)):
                if self.data.levels == self.levels_max + 1:
                        warnings.warn("Failed to achieve weak convergence. levels == levels_max.", MaxLevelsWarning)
                # add another level
                self.data._add_level()
            else:
                # both conditions met
                break
            total_next_samples = (self.data.replications*self.data.eval_level*self.data.n_level*2).sum()
            if (self.data.n_total + total_next_samples) > self.n_max:
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples, which would exceed n_max = %d.
                Stopping integration process.
                Note that error tolerances may no longer be satisfied""" \
                % (int(self.data.n_total), int(total_next_samples), int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
        self.data.time_integrate = time() - t_start
        return self.data.solution,self.data
    
    def set_tolerance(self, abs_tol=None, alpha=.01, rmse_tol=None):
        """
        See abstract method. 
        
        Args:
            integrand (Integrand): integrand with multi-level g method
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            alpha (float): uncertaintly level.
                If rmse_tol not supplied, then rmse_tol = abs_tol/norm.ppf(1-alpha/2)
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not.
                Takes priority over aboluste tolerance and alpha if supplied. 
        """
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = (float(abs_tol) / norm.ppf(1-alpha/2.))

