from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import LDTransformData
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
from ..integrand import Integrand
from numpy import *
from time import time
import warnings


class _CubQMCLDG(StoppingCriterion):
    """
    Abstract class for CubQMC{LD}G where LD is a low discrepancy discrete distribution. 
    See subclasses for implementation differences for each LD sequence. 
    """

    def __init__(self, integrand, abs_tol, rel_tol, n_init, n_max, fudge, check_cone,
        control_variates, control_variate_means, update_beta, ptransform,
        coefv, allowed_levels, allowed_distribs, cast_complex, error_fun):
        self.parameters = ['abs_tol','rel_tol','n_init','n_max']
        # Input Checks
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        m_min = log2(n_init)
        m_max = log2(n_max)
        if m_min%1 != 0. or m_min < 8. or m_max%1 != 0.:
            warning_s = '''
                n_init and n_max must be a powers of 2.
                n_init must be >= 2^8.
                Using n_init = 2^10 and n_max=2^35.'''
            warnings.warn(warning_s, ParameterWarning)
            m_min = 10.
            m_max = 35.
        self.n_init = 2.**m_min
        self.n_max = 2.**m_max
        self.m_min = m_min
        self.m_max = m_max
        self.fudge = fudge
        self.check_cone = check_cone
        self.coefv = coefv
        self.ptransform = ptransform
        self.cast_complex = cast_complex
        self.error_fun = error_fun
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.dprime = self.integrand.dprime
        self.cv = list(atleast_1d(control_variates))
        self.ncv = len(self.cv)
        self.cv_mu = atleast_2d(control_variate_means)
        if self.cv_mu.size!=(self.dprime*self.ncv):
            raise ParameterError('''Control variate means should have shape (dprime,len(control variates)).''')
        self.cv_mu = self.cv_mu.reshape(self.dprime,self.ncv)
        for cv in self.cv:
            if (cv.discrete_distrib!=self.discrete_distrib) or (not isinstance(cv,Integrand)) or (cv.dprime!=self.dprime):
                raise ParameterError('''
                        Each control variate's discrete distribution must be an Integrand instance 
                        with the same discrete distribution as the main integrand. dprime must also match 
                        that of the main integrand instance for each control variate.''')
        self.update_beta = update_beta
        if self.ncv>0:
            self.parameters += ['cv','cv_mu','update_beta']
        super(_CubQMCLDG,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals=True)

    def integrate(self):
        t_start = time()
        self.datum = [LDTransformData(self.m_min,self.m_max,self.coefv,self.fudge,self.check_cone,self.ncv,self.cv_mu[j],self.update_beta) for j in range(self.dprime)]
        self.data = LDTransformData.__new__(LDTransformData)
        self.data.flags_indv = tile(True,self.dprime)
        self.data.m = tile(self.m_min,self.dprime)
        self.data.n_min = 0
        self.data.bounds = vstack([tile(-inf,(1,self.dprime)),tile(inf,(1,self.dprime))])
        self.data.solution_indv = tile(nan,self.dprime)
        self.data.solution = nan
        while True:
            m = self.data.m.max()
            n_min = self.data.n_min
            n_max = int(2**m)
            n = int(n_max-n_min)
            x = self.discrete_distrib.gen_samples(n_min=n_min,n_max=n_max)
            ycvfull = zeros((1+self.ncv,n,self.dprime),dtype=float)
            ycvfull[0] = self.integrand.f(x,periodization_transform=self.ptransform,compute_flags=self.data.flags_indv)
            for k in range(self.ncv):
                ycvfull[1+k] = self.cv[k].f(x,periodization_transform=self.ptransform,compute_flags=self.data.flags_indv)
            ycvfull_cp = ycvfull.astype(complex) if self.cast_complex else ycvfull.copy()
            for j in range(self.dprime):
                if not self.data.flags_indv[j]: continue
                y_val = ycvfull[0,:,j]
                y_cp = ycvfull_cp[0,:,j]
                yg_val = ycvfull[1:,:,j].T
                yg_cp = ycvfull_cp[1:,:,j].T
                self.data.solution_indv[j],self.data.bounds[:,j],cone_violation = self.datum[j].update_data(m,y_val,y_cp,yg_val,yg_cp)
                if cone_violation:
                    warnings.warn('Function at index %d (indexing dprime) violates cone conditions.'%j,CubatureWarning)
            self.data.indv_error = (self.data.bounds[1]-self.data.bounds[0])/2
            self.data.ci_low,self.data.ci_high = self.data.bounds[0],self.data.bounds[1]
            self.data.ci_comb_low,self.data.ci_comb_high,self.data.violated = self.integrand.bound_fun(self.data.ci_low,self.data.ci_high)
            error_low = self.error_fun(self.data.ci_comb_low,self.abs_tol,self.rel_tol)
            error_high = self.error_fun(self.data.ci_comb_high,self.abs_tol,self.rel_tol)
            self.data.solution = 1/2*(self.data.ci_comb_low+self.data.ci_comb_high+error_low-error_high)
            rem_error_low = abs(self.data.ci_comb_low-self.data.solution)-error_low
            rem_error_high = abs(self.data.ci_comb_high-self.data.solution)-error_high
            self.data.flags_comb = maximum(rem_error_low,rem_error_high)>=0
            self.data.flags_comb |= self.data.violated
            self.data.flags_indv = self.integrand.dependency(self.data.flags_comb)
            self.data.n = 2**m
            self.data.n_total = self.data.n.max()
            if sum(self.data.flags_indv)==0:
                break # stopping criterion met
            elif 2*self.data.n_total>self.n_max:
                # doubling samples would go over n_max
                warning_s = """
                Alread generated %d samples.
                Trying to generate %d new samples would exceed n_max = %d.
                No more samples will be generated.
                Note that error tolerances may no longer be satisfied.""" \
                % (int(self.data.n_total),int(self.data.n_total),int(self.n_max))
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
        self.data.time_integrate = time()-t_start
        return self.data.solution,self.data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol != None: self.abs_tol = abs_tol
        if rel_tol != None: self.rel_tol = rel_tol
