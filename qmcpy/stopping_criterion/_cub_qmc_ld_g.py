from ._stopping_criterion import StoppingCriterion
from ..accumulate_data import LDTransformData
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
from ..integrand import Integrand
import numpy as np
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
        m_min = np.log2(n_init)
        m_max = np.log2(n_max)
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
        self.d = self.discrete_distrib.d
        self.d_indv = self.integrand.d_indv
        self.cv = list(np.atleast_1d(control_variates))
        self.ncv = len(self.cv)
        self.cv_mu = np.array(control_variate_means) if self.ncv>0 else np.empty((self.ncv,)+self.d_indv)
        self.cv_mu = self.cv_mu if self.cv_mu.ndim>1 else self.cv_mu.reshape(self.ncv,-1)
        if self.cv_mu.shape!=((self.ncv,)+self.d_indv):
            raise ParameterError('''Control variate means should have shape (len(control variates),d_indv).''')
        for cv in self.cv:
            if (cv.discrete_distrib!=self.discrete_distrib) or (not isinstance(cv,Integrand)) or (cv.d_indv!=self.d_indv):
                raise ParameterError('''
                        Each control variates discrete distribution must be an Integrand instance 
                        with the same discrete distribution as the main integrand. d_indv must also match 
                        that of the main integrand instance for each control variate.''')
        self.update_beta = update_beta
        if self.ncv>0:
            self.parameters += ['cv','cv_mu','update_beta']
        super(_CubQMCLDG,self).__init__(allowed_levels, allowed_distribs, allow_vectorized_integrals=True)

    def integrate(self):
        t_start = time()
        self.datum = np.empty(self.d_indv,dtype=object)
        for j in np.ndindex(self.d_indv):
            cv_mu_j = self.cv_mu[(slice(None),)+j]
            self.datum[j] = LDTransformData(self.m_min,self.m_max,self.coefv,self.fudge,self.check_cone,self.ncv,cv_mu_j,self.update_beta)
        self.data = LDTransformData.__new__(LDTransformData)
        self.data.flags_indv = np.tile(False,self.d_indv)
        self.data.compute_flags = np.tile(True,self.d_indv)
        self.data.m = np.tile(self.m_min,self.d_indv)
        self.data.n_min = 0
        self.data.indv_bound_low = np.tile(-np.inf,self.d_indv)
        self.data.indv_bound_high = np.tile(np.inf,self.d_indv)
        self.data.solution_indv = np.tile(np.nan,self.d_indv)
        self.data.solution = np.nan
        self.data.xfull = np.empty((0,self.d))
        self.data.yfull = np.empty((0,)+self.d_indv)
        while True:
            m = self.data.m.max()
            n_min = self.data.n_min
            n_max = int(2**m)
            n = int(n_max-n_min)
            xnext = self.discrete_distrib.gen_samples(n_min=n_min,n_max=n_max)
            ycvnext = np.empty((1+self.ncv,n,)+self.d_indv,dtype=float)
            ycvnext[0] = self.integrand.f(xnext,periodization_transform=self.ptransform,compute_flags=self.data.compute_flags)
            for k in range(self.ncv):
                ycvnext[1+k] = self.cv[k].f(xnext,periodization_transform=self.ptransform,compute_flags=self.data.compute_flags)
            ycvnext_cp = ycvnext.astype(complex) if self.cast_complex else ycvnext.copy()
            for j in np.ndindex(self.d_indv):
                if self.data.flags_indv[j]: continue
                slice_yj = (0,slice(None),)+j
                slice_ygj = (slice(1,None),slice(None),)+j
                y_val = ycvnext[slice_yj]
                y_cp = ycvnext_cp[slice_yj]
                yg_val = ycvnext[slice_ygj].T
                yg_cp = ycvnext_cp[slice_ygj].T
                self.data.solution_indv[j],self.data.indv_bound_low[j],self.data.indv_bound_high[j],cone_violation = self.datum[j].update_data(m,y_val,y_cp,yg_val,yg_cp)
                if cone_violation:
                    warnings.warn('Function at index %d (indexing d_indv) violates cone conditions.'%j,CubatureWarning)
            self.data.xfull = np.vstack((self.data.xfull,xnext))
            self.data.yfull = np.vstack((self.data.yfull,ycvnext[0]))
            self.data.comb_bound_low,self.data.comb_bound_high = self.integrand.bound_fun(self.data.indv_bound_low,self.data.indv_bound_high)
            self.abs_tols,self.rel_tols = np.full_like(self.data.comb_bound_low,self.abs_tol),np.full_like(self.data.comb_bound_low,self.rel_tol)
            fidxs = np.isfinite(self.data.comb_bound_low)&np.isfinite(self.data.comb_bound_high)
            slow,shigh,abs_tols,rel_tols = self.data.comb_bound_low[fidxs],self.data.comb_bound_high[fidxs],self.abs_tols[fidxs],self.rel_tols[fidxs]
            self.data.solution = np.tile(np.nan,self.data.comb_bound_low.shape)
            self.data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            self.data.comb_flags = np.tile(False,self.data.comb_bound_low.shape)
            self.data.comb_flags[fidxs] = (shigh-slow) < (self.error_fun(slow,abs_tols,rel_tols)+self.error_fun(shigh,abs_tols,rel_tols))
            self.data.flags_indv = self.integrand.dependency(self.data.comb_flags)
            self.data.compute_flags = ~self.data.flags_indv
            self.data.n = 2**self.data.m
            self.data.n_total = self.data.n.max()
            if np.sum(self.data.compute_flags)==0:
                break # stopping criterion met
            elif 2*self.data.n_total>self.n_max:
                # doubling samples would go over n_max
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceed n_max = %d.
                No more samples will be generated.
                Note that error tolerances may no longer be satisfied.""" \
                % (int(self.data.n_total),int(self.data.n_total),int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                self.data.n_min = n_max
                self.data.m += self.data.compute_flags
        self.data.integrand = self.integrand
        self.data.true_measure = self.true_measure
        self.data.discrete_distrib = self.discrete_distrib
        self.data.stopping_crit = self
        self.data.parameters = [
            'solution',
            'comb_bound_low',
            'comb_bound_high',
            'comb_flags',
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
