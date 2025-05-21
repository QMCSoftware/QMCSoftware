from .abstract_stopping_criterion import AbstractStoppingCriterion
from ..accumulate_data import AccumulateData
from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
from ..integrand import AbstractIntegrand
import numpy as np
from time import time
import warnings


class _CubQMCLDG(AbstractStoppingCriterion):
    """
    Abstract class for CubQMC{LD}G where LD is a low discrepancy discrete distribution. 
    See subclasses for implementation differences for each LD sequence. 
    """

    def __init__(self, integrand, abs_tol, rel_tol, n_init, n_limit, fudge, check_cone,
        control_variates, control_variate_means, update_beta, ptransform,
        ft, omega, allowed_levels, allowed_distribs, cast_complex, error_fun):
        self.parameters = ['abs_tol','rel_tol','n_init','n_limit']
        # Input Checks
        if np.log2(n_init)%1!=0 or n_init<2**8:
            warnings.warn('n_init must be a power of two at least 2**8. Using n_init = 2**8',ParameterWarning)
            n_init = 2**8
        if np.log2(n_limit)%1!=0:
            warnings.warn('n_init must be a power of two. Using n_limit = 2**30',ParameterWarning)
            n_limit = 2**30
        # Set Attributes
        self.n_init = int(n_init)
        self.m_init = int(np.log2(n_init))
        self.n_limit = int(n_limit)
        self.error_fun = error_fun
        self.fudge = fudge
        self.check_cone = check_cone
        self.ft = ft
        self.omega = omega
        self.ptransform = ptransform
        self.cast_complex = cast_complex
        self.r_lag = 4
        self.omg_circ = lambda m: 2**(-m)
        self.l_star = int(self.m_init-self.r_lag)
        self.omg_hat = lambda m: self.fudge(m)/((1+self.fudge(self.r_lag))*self.omg_circ(self.r_lag))
        # QMCPy Objs
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.true_measure.discrete_distrib
        super(_CubQMCLDG,self).__init__(allowed_levels=allowed_levels,allowed_distribs=allowed_distribs,allow_vectorized_integrals=True)
        assert self.integrand.discrete_distrib.no_replications==True, "Require the discrete distribution has replications=None"
        assert self.integrand.discrete_distrib.randomize!="FALSE", "Require discrete distribution is randomized"
        self.set_tolerance(abs_tol,rel_tol)
        # control variates
        self.cv_mu = np.atleast_1d(control_variate_means)
        self.cv = control_variates
        if isinstance(self.cv,AbstractIntegrand):
            self.cv = [self.cv]
            self.cv_mu = self.cv_mu[None,...]
        assert isinstance(self.cv,list), "cv must be a list of AbstractIntegrand objects"
        for cv in self.cv:
            if (not isinstance(cv,AbstractIntegrand)) or (cv.discrete_distrib!=self.discrete_distrib) or (cv.d_indv!=self.d_indv):
                raise ParameterError('''
                        Each control variates discrete distribution must be an AbstractIntegrand instance 
                        with the same discrete distribution as the main integrand. d_indv must also match 
                        that of the main integrand instance for each control variate.''')
        self.ncv = len(self.cv)
        self.update_beta = update_beta
        if self.ncv>0:
            assert self.cv_mu.shape==((self.ncv,)+self.integrand.d_indv), "Control variate means should have shape (len(control variates),d_indv)."
            self.parameters += ['cv','cv_mu','update_beta']
        else:
            self.update_beta = False
    
    def _update_kappanumap(self, kappanumap, ytildefull, mfrom, mto, m):
        for l in range(int(mfrom),int(mto),-1):
            nl = 2**l
            oldone = np.abs(np.take_along_axis(ytildefull,kappanumap[...,1:int(nl)],axis=-1)) # earlier values of kappa, don't touch first one
            newone = np.abs(np.take_along_axis(ytildefull,kappanumap[...,nl+1:2*nl],axis=-1)) # later values of kappa,
            *prioridxs,flip = np.where(newone>oldone)
            flip = flip+1 # add one to account for the fact that we do not consider indices which are powers of 2
            if flip.size!=0:
                additive = np.arange(0,2**m-1,2**(l+1))
                flipall = (flip[None,:]+additive[:,None]).flatten()
                zeroadditive = np.zeros(len(additive),dtype=int) 
                pidxs = [(pidx[None,:]+zeroadditive[:,None]).flatten() for pidx in prioridxs] # alternative to tiling
                kappanumap[*pidxs,flipall],kappanumap[*pidxs,nl+flipall] = kappanumap[*pidxs,nl+flipall],kappanumap[*pidxs,flipall]
        return kappanumap
    def integrate(self):
        t_start = time()
        data = LDTransformAccumulateData(
            parameters = [
                'solution',
                'comb_bound_low',
                'comb_bound_high',
                'comb_bound_diff',
                'comb_flags',
                'n_total',
                'n',
                'time_integrate'])
        data.flags_indv = np.tile(False,self.integrand.d_indv)
        data.compute_flags = np.tile(True,self.integrand.d_indv)
        data.n = np.tile(self.n_init,self.integrand.d_indv)
        data.n_min = 0 
        data.n_max = self.n_init
        data.solution_indv = np.tile(np.nan,self.integrand.d_indv)
        data.xfull = np.empty((0,self.integrand.d))
        data.yfull = np.empty(self.integrand.d_indv+(0,))
        data.ycvfull = np.empty(self.integrand.d_indv+(self.ncv,0))
        data.bounds_half_width = np.tile(np.inf,self.integrand.d_indv)
        data.muhat = np.tile(np.nan,self.integrand.d_indv)
        while True:
            m = int(np.log2(data.n_max))
            xnext = self.discrete_distrib.gen_samples(n_min=data.n_min,n_max=data.n_max)
            data.xfull = np.concatenate([data.xfull,xnext],0)
            ynext = self.integrand.f(xnext,periodization_transform=self.ptransform,compute_flags=data.compute_flags)
            ynext[~data.compute_flags] = np.nan
            data.yfull = np.concatenate([data.yfull,ynext],-1)
            if self.ncv>0:
                ycvnext = [None]*self.ncv
                for k in range(self.ncv):
                    ycvnext_k = self.cv[k].f(xnext,periodization_transform=self.ptransform,compute_flags=data.compute_flags)
                    ycvnext_k[~data.compute_flags] = np.nan
                    ycvnext[k] = ycvnext_k
                ycvnext = np.stack(ycvnext,-2)
                data.ycvfull = np.concatenate([data.ycvfull,ycvnext],-1)
            mllstart = m-self.r_lag-1
            nllstart = int(2**mllstart)
            if data.n_min==0: # first iteration
                n = int(2**m)
                data.ytildefull = self.ft(ynext)/np.sqrt(n)
                data.kappanumap = np.tile(np.arange(n),self.integrand.d_indv+(1,))
                data.kappanumap = self._update_kappanumap(data.kappanumap,data.ytildefull,m-1,0,m)
                if self.ncv>0:
                    data.ycvtildefull = self.ft(ycvnext)/np.sqrt(n)
                    self.beta_update(mllstart)
                    data.kappanumap = np.tile(np.arange(n),self.integrand.d_indv+(1,))
                    data.kappanumap = self._update_kappanumap(data.kappanumap,data.ytildefull,m-1,0,m)
            else: # any iteration after the first
                mnext = int(m-1)
                n = int(2**mnext)
                if self.ncv>0:
                    self.y_val[-n:] = self.y_val[-n:]-self.yg_val[-n:]@self.beta
                    self.y_cp[-n:] = self.y_val[-n:]
                if not self.update_beta: # do not update the beta coefficients
                    ytildeomega = self.omega(mnext)*self.ft(ynext[data.compute_flags])/np.sqrt(n)
                    ytildefull_next = np.nan*np.ones_like(data.ytildefull)
                    ytildefull_next[data.compute_flags] = (data.ytildefull[data.compute_flags]-ytildeomega)/2
                    data.ytildefull[data.compute_flags] = (data.ytildefull[data.compute_flags]+ytildeomega)/2
                    data.ytildefull = np.concatenate([data.ytildefull,ytildefull_next],axis=-1)
                    data.kappanumap = np.concatenate([data.kappanumap,n+data.kappanumap],axis=-1)
                    data.kappanumap[data.compute_flags] = self._update_kappanumap(data.kappanumap[data.compute_flags],data.ytildefull[data.compute_flags],m-1,mllstart,m)
                else: # update beta
                    self.y_cp = self.fast_transform(self.y_cp,0,m,m)
                    self.yg_cp = self.fast_transform(self.yg_cp,0,m,m)
                    self.beta_update(mllstart)
                    data.kappanumap = np.hstack((data.kappanumap,2**(m-1)+data.kappanumap)).astype(int)
                    data.kappanumap[data.compute_flags] = self._update_kappanumap(data.kappanumap[data.compute_flags],data.ytildefull[data.compute_flags],m-1,mllstart,m)
            data.muhat[data.compute_flags] = data.yfull[data.compute_flags].mean(-1)+data.beta@self.cv_mu if self.ncv>0 else data.yfull[data.compute_flags].mean(-1)
            data.bounds_half_width[data.compute_flags] = self.fudge(m)*np.abs(np.take_along_axis(data.ytildefull[data.compute_flags],data.kappanumap[data.compute_flags][...,nllstart:2*nllstart],axis=-1)).sum(-1)
            data.indv_bound_low = data.muhat-data.bounds_half_width
            data.indv_bound_high = data.muhat+data.bounds_half_width
            if self.check_cone:
                data.c_stilde_low = np.tile(-np.inf,(m+1-self.l_star,)+self.integrand.d_indv)
                data.c_stilde_up = np.tile(np.inf,(m+1-self.l_star,)+self.integrand.d_indv)
                for l in range(self.l_star,m+1): # Storing the information for the necessary conditions
                    c_tmp = self.omg_hat(m-l)*self.omg_circ(m-l)
                    c_low = 1./(1+c_tmp)
                    c_up = 1./(1-c_tmp)
                    const1 = np.abs(np.take_along_axis(data.ytildefull[data.compute_flags],data.kappanumap[data.compute_flags][...,int(2**(l-1)):int(2**l)],axis=-1)).sum(-1)
                    idx = int(l-self.l_star)
                    data.c_stilde_low[idx,data.compute_flags] = np.maximum(data.c_stilde_low[idx,data.compute_flags],c_low*const1)
                    if c_tmp < 1:
                        data.c_stilde_up[idx,data.compute_flags] = np.minimum(data.c_stilde_up[idx,data.compute_flags],c_up*const1)
                data.cone_violation = (data.c_stilde_low > data.c_stilde_up).any(0)
                if data.cone_violation.sum()>0:
                    warnings.warn('Cone condition violations at indices in data.cone_violation.',CubatureWarning)
            else:
                data.cone_violation = None
            data.n[data.compute_flags] = data.n_max
            data.n_total = data.n_max
            data.comb_bound_low,data.comb_bound_high = self.integrand.bound_fun(data.indv_bound_low,data.indv_bound_high)
            data.comb_bound_diff = data.comb_bound_high-data.comb_bound_low
            fidxs = np.isfinite(data.comb_bound_low)&np.isfinite(data.comb_bound_high)
            slow,shigh,abs_tols,rel_tols = data.comb_bound_low[fidxs],data.comb_bound_high[fidxs],self.abs_tols[fidxs],self.rel_tols[fidxs]
            data.solution = np.tile(np.nan,data.comb_bound_low.shape)
            data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            data.comb_flags = np.tile(False,data.comb_bound_low.shape)
            data.comb_flags[fidxs] = (shigh-slow) <= (self.error_fun(slow,abs_tols,rel_tols)+self.error_fun(shigh,abs_tols,rel_tols))
            data.flags_indv = self.integrand.dependency(data.comb_flags)
            data.compute_flags = ~data.flags_indv
            if np.sum(data.compute_flags)==0:
                break # sufficiently estimated
            elif 2*data.n_total>self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceeds n_limit = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied. """ \
                % (int(data.n_total),int(data.n_total),int(self.n_limit))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            data.n_min = data.n_max
            data.n_max = 2*data.n_min
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None):
        """
        See abstract method. 
        
        Args:
            abs_tol (float): absolute tolerance. Reset if supplied, ignored if not. 
            rel_tol (float): relative tolerance. Reset if supplied, ignored if not. 
        """
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb,self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb,self.rel_tol)

class LDTransformAccumulateData(AccumulateData):

    
    def beta_update(self, mstart):
        kappa_approx = self.kappanumap[int(2**mstart):] # kappa index used for fitting
        x4beta = self.ycvtildefull[...,kappa_approx]
        y4beta = data.ytildefull[...,kappa_approx]
        self.beta = np.linalg.lstsq(x4beta,y4beta,rcond=None)[0]
        self.yfull = self.yfull-(self.ycvfull*self.beta[...,None]).sum(-2) # get new function values
        data.ytildefull = data.ytildefull-(self.ycvtildefull*self.beta[...,None]).sum(-2) # redefine function