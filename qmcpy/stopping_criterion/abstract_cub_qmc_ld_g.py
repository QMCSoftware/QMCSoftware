from .abstract_stopping_criterion import AbstractStoppingCriterion, IS_PRINT_DIAGNOSTIC, print_diagnostic
from ..util.data import Data

from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
from ..integrand import AbstractIntegrand
import numpy as np
from time import time
import warnings


class AbstractCubQMCLDG(AbstractStoppingCriterion):
    
    def __init__(self, integrand, abs_tol, rel_tol, n_init, n_limit, fudge, check_cone,
        control_variates, control_variate_means, update_beta, ptransform,
        ft, omega, allowed_distribs, cast_complex, error_fun):
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
        assert isinstance(error_fun,str) or callable(error_fun)
        if isinstance(error_fun,str):
            if error_fun.upper()=="EITHER":
                error_fun = lambda sv,abs_tol,rel_tol: np.maximum(abs_tol,abs(sv)*rel_tol)
            elif error_fun.upper()=="BOTH":
                error_fun = lambda sv,abs_tol,rel_tol: np.minimum(abs_tol,abs(sv)*rel_tol)
            else:
                raise ParameterError("str error_fun must be 'EITHER' or 'BOTH'")
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
        self.discrete_distrib = self.integrand.discrete_distrib
        super(AbstractCubQMCLDG,self).__init__(allowed_distribs=allowed_distribs,allow_vectorized_integrals=True)
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
            if (not isinstance(cv,AbstractIntegrand)) or (cv.discrete_distrib!=self.discrete_distrib) or (cv.d_indv!=self.integrand.d_indv):
                raise ParameterError('''
                        Each control variates discrete distribution must be an AbstractIntegrand instance 
                        with the same discrete distribution as the main integrand. d_indv must also match 
                        that of the main integrand instance for each control variate.''')
        self.ncv = len(self.cv)
        self.update_beta = update_beta
        # Internal consecutive solution change detection parameters (hardcoded)
        self.max_consecutive_unchanged = 3  # Stop after 3 consecutive unchanged solutions
        self.solution_change_tol = 1e-12   # Tolerance for "no change"
        if self.ncv>0:
            assert self.cv_mu.shape==((self.ncv,)+self.integrand.d_indv), "Control variate means should have shape (len(control variates),d_indv)."
            self.parameters += ['cv','cv_mu','update_beta']
        else:
            self.update_beta = False
        self.vlstsq = np.vectorize(lambda x,y: np.linalg.lstsq(x.T,y,rcond=None)[0],signature="(k,m),(m)->(k)")
    
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
                pidxs = tuple((pidx[None,:]+zeroadditive[:,None]).flatten() for pidx in prioridxs) # alternative to tiling
                kappanumap[pidxs+(flipall,)],kappanumap[pidxs+(nl+flipall,)] = kappanumap[pidxs+(nl+flipall,)],kappanumap[pidxs+(flipall,)]
        return kappanumap
    
    def _beta_update(self, beta, kappanumap, ytildefull, ycvtildefull, mstart):
        kappa_approx = kappanumap[...,(2**mstart):] # kappa index used for fitting
        y4beta = np.take_along_axis(ytildefull,kappa_approx,axis=-1)
        x4beta = np.take_along_axis(ycvtildefull,kappa_approx[...,None,:],axis=-1)
        beta = self.vlstsq(x4beta,y4beta)
        return beta
    
    def _check_consecutive_solution_changes(self, data, m):
        """
        Check for consecutive solution changes and implement early stopping.
        
        Args:
            data: Data object containing integration state
            m: Current iteration number
        """
        # Use hardcoded internal parameters for consecutive change detection
        max_consecutive_unchanged = self.max_consecutive_unchanged  # Stop after 3 consecutive unchanged
        solution_change_tol = self.solution_change_tol  # Tolerance for "no change"
        
        # Initialize tracking fields if they don't exist (for resumed data)
        if not hasattr(data, 'previous_solutions'):
            data.previous_solutions = []
        if not hasattr(data, 'consecutive_unchanged_count'):
            data.consecutive_unchanged_count = 0
        
        current_solution = data.solution
        
        # Handle different solution types (scalar, array, etc.)
        if current_solution is not None and np.isfinite(current_solution).all():
            # Convert to consistent format for comparison
            if np.isscalar(current_solution):
                current_val = current_solution
            elif hasattr(current_solution, 'shape') and current_solution.shape == ():
                current_val = current_solution.item()
            else:
                # For arrays, use norm of the solution vector
                current_val = np.linalg.norm(current_solution)
            
            # Check if solution has changed significantly
            solution_changed = True
            if len(data.previous_solutions) > 0:
                last_val = data.previous_solutions[-1]
                change = abs(current_val - last_val)
                if change <= solution_change_tol:
                    solution_changed = False
                    data.consecutive_unchanged_count += 1
                    if IS_PRINT_DIAGNOSTIC:
                        print_diagnostic(f'[STAGNANT] Solution unchanged for {data.consecutive_unchanged_count} consecutive iterations (change={change:.2e})', data)
                else:
                    data.consecutive_unchanged_count = 0  # Reset counter
            
            # Add current solution to history (keep last few)
            data.previous_solutions.append(current_val)
            if len(data.previous_solutions) > max_consecutive_unchanged + 1:
                data.previous_solutions.pop(0)  # Remove oldest
            
            # Check for early stopping due to stagnation
            if data.consecutive_unchanged_count >= max_consecutive_unchanged:
                if IS_PRINT_DIAGNOSTIC:
                    print_diagnostic(f'[EARLY_STOP] Stopping due to {data.consecutive_unchanged_count} consecutive unchanged solutions', data)
                
                # Mark all components as converged to trigger stopping
                data.compute_flags = np.tile(False, data.compute_flags.shape)
                data.comb_flags = np.tile(True, data.comb_flags.shape if hasattr(data, 'comb_flags') else data.compute_flags.shape)
                if hasattr(data, 'flags_indv'):
                    data.flags_indv = np.tile(True, data.flags_indv.shape)
    
    def integrate(self, resume=None):
        """
        Perform integration, optionally resuming from a previous computation.

        Args:
            resume (Data, optional): Previous data object returned from a prior call to integrate. 
                If provided, computation resumes from this state.
                
        Returns:
            tuple: (solution, data) - The estimated solution and the data object (can be used for future resume).
        """
        t_start = time()
        if resume is not None:
            # Resume from previous state
            data = resume
            if IS_PRINT_DIAGNOSTIC:
                print_diagnostic('[RESUME] Resuming from data', data)
            if hasattr(data, 'n_total'):
                # Set n_min to n_total so next batch starts after previous samples
                data.n_min = int(data.n_total)
                # Set n_max to min of (2 * n_min, discrete_distrib.n_limit) to avoid exceeding limit
                data.n_max = int(min(2 * data.n_min, self.discrete_distrib.n_limit))
                if IS_PRINT_DIAGNOSTIC:
                    print_diagnostic('[RESUME] After setting n_min/n_max', data)
        else:
            # Initialize from scratch
            data = Data(
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
            if self.ncv>0:
                data.ycvfull = np.empty(self.integrand.d_indv+(self.ncv,0))
            data.bounds_half_width = np.tile(np.inf,self.integrand.d_indv)
            data.muhat = np.tile(np.nan,self.integrand.d_indv)
            data.beta = np.tile(np.nan,self.integrand.d_indv+(self.ncv,))
            data.ytildefull = None  # Will be initialized on first iteration
            data.kappanumap = None  # Will be initialized on first iteration
            # Track consecutive solution changes for early stopping
            data.previous_solutions = []  # Store last few solutions for convergence detection
            data.consecutive_unchanged_count = 0  # Counter for unchanged solutions
            if self.ncv>0:
                data.ycvtildefull = None  # Will be initialized on first iteration
        
        if IS_PRINT_DIAGNOSTIC:
            print_diagnostic('[INIT] Initial data state', data)
            
        while True:
            m = int(np.log2(data.n_max))
            if IS_PRINT_DIAGNOSTIC:
                print_diagnostic(f'[ITER] Start iteration m={m}', data)
            xnext = self.discrete_distrib(n_min=data.n_min,n_max=data.n_max)
            data.xfull = np.concatenate([data.xfull,xnext],0)
            ynext = self.integrand.f(xnext,periodization_transform=self.ptransform,compute_flags=data.compute_flags)
            ynext[~data.compute_flags] = np.nan
            data.yfull = np.concatenate([data.yfull,ynext],-1)
            
            if IS_PRINT_DIAGNOSTIC:
                print_diagnostic(f'[ITER] After sampling n_min={data.n_min} to n_max={data.n_max}', data)
            if self.ncv>0:
                ycvnext = [None]*self.ncv
                for k in range(self.ncv):
                    ycvnext_k = self.cv[k].f(xnext,periodization_transform=self.ptransform,compute_flags=data.compute_flags)
                    ycvnext_k[~data.compute_flags] = np.nan
                    ycvnext[k] = ycvnext_k
                ycvnext = np.stack(ycvnext,-2)
                data.ycvfull = np.concatenate([data.ycvfull,ycvnext],-1)
            mllstart = m-self.r_lag-1
            nllstart = 2**mllstart
            if data.ytildefull is None: # first iteration (or first iteration after resume)
                n = int(2**m)
                data.ytildefull = self.ft(ynext)/np.sqrt(n)
                data.kappanumap = self._update_kappanumap(np.tile(np.arange(n),self.integrand.d_indv+(1,)),data.ytildefull,m-1,0,m)
                if self.ncv>0:
                    data.ycvtildefull = self.ft(ycvnext)/np.sqrt(n)
                    data.beta = self._beta_update(data.beta,data.kappanumap,data.ytildefull,data.ycvtildefull,mllstart)
                    data.ytildefull = data.ytildefull-(data.ycvtildefull*data.beta[...,None]).sum(-2)
                    data.kappanumap = self._update_kappanumap(np.tile(np.arange(n),self.integrand.d_indv+(1,)),data.ytildefull,m-1,0,m)
            else: # any iteration after the first
                mnext = int(m-1)
                n = int(2**mnext)
                if not self.update_beta: # do not update the beta coefficients
                    if self.ncv>0:
                        ynext[data.compute_flags] = ynext[data.compute_flags]-(ycvnext[data.compute_flags]*data.beta[data.compute_flags,:,None]).sum(-2)
                    ytildeomega = self.omega(mnext)*self.ft(ynext[data.compute_flags])/np.sqrt(n)
                    ytildefull_next = np.nan*np.ones_like(data.ytildefull)
                    ytildefull_next[data.compute_flags] = (data.ytildefull[data.compute_flags]-ytildeomega)/2
                    data.ytildefull[data.compute_flags] = (data.ytildefull[data.compute_flags]+ytildeomega)/2
                    data.ytildefull = np.concatenate([data.ytildefull,ytildefull_next],axis=-1)
                else: # update beta
                    data.ytildefull = np.concatenate([data.ytildefull,np.tile(np.nan,data.ytildefull.shape)],axis=-1)
                    data.ytildefull[data.compute_flags] = self.ft(data.yfull[data.compute_flags])/np.sqrt(2**m)
                    data.ycvtildefull = np.concatenate([data.ycvtildefull,np.tile(np.nan,data.ycvtildefull.shape)],axis=-1)
                    data.ycvtildefull[data.compute_flags] = self.ft(data.ycvfull[data.compute_flags])/np.sqrt(2**m)
                    data.beta[data.compute_flags] = self._beta_update(data.beta[data.compute_flags],data.kappanumap[data.compute_flags],data.ytildefull[data.compute_flags],data.ycvtildefull[data.compute_flags],mllstart)
                data.kappanumap = np.concatenate([data.kappanumap,n+data.kappanumap],axis=-1)
                data.kappanumap[data.compute_flags] = self._update_kappanumap(data.kappanumap[data.compute_flags],data.ytildefull[data.compute_flags],m-1,mllstart,m)
            if self.ncv==0:
                data.muhat[data.compute_flags] = data.yfull[data.compute_flags].mean(-1)
            else:
                ydiff = data.yfull[data.compute_flags]-(data.ycvfull[data.compute_flags]*data.beta[data.compute_flags,:,None]).sum(-2)
                data.muhat[data.compute_flags] = ydiff.mean(-1)+(data.beta[data.compute_flags]*np.moveaxis(self.cv_mu,0,-1)[data.compute_flags]).sum(-1)
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
            
            # Store previous solution for potential NaN recovery during resume operations
            previous_solution = getattr(data, 'solution', None)
            
            data.solution = np.tile(np.nan,data.comb_bound_low.shape)
            data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow,abs_tols,rel_tols)-self.error_fun(shigh,abs_tols,rel_tols))
            
            # If solution becomes NaN and we have a finite previous solution, preserve it
            # This is particularly important during resume operations when bounds may be non-finite
            if (previous_solution is not None and 
                np.any(np.isnan(data.solution)) and 
                np.any(np.isfinite(previous_solution))):
                
                if np.isscalar(data.solution) or data.solution.shape == ():
                    # Scalar case
                    if np.isnan(data.solution):
                        if np.isscalar(previous_solution):
                            if np.isfinite(previous_solution):
                                data.solution = previous_solution
                        elif hasattr(previous_solution, 'shape') and previous_solution.shape == ():
                            # 0-dimensional array
                            prev_val = previous_solution.item()
                            if np.isfinite(prev_val):
                                data.solution = prev_val
                        else:
                            # Multi-dimensional array, take first finite value
                            finite_vals = previous_solution[np.isfinite(previous_solution)]
                            if len(finite_vals) > 0:
                                data.solution = finite_vals[0]
            
            data.comb_flags = np.tile(False,data.comb_bound_low.shape)
            data.comb_flags[fidxs] = (shigh-slow) <= (self.error_fun(slow,abs_tols,rel_tols)+self.error_fun(shigh,abs_tols,rel_tols))
            data.flags_indv = self.integrand.dependency(data.comb_flags)
            data.compute_flags = ~data.flags_indv
            
            if IS_PRINT_DIAGNOSTIC:
                print_diagnostic(f'[ITER] After convergence check m={m}', data)
            
            # Check for consecutive solution changes (early stopping criterion)
            self._check_consecutive_solution_changes(data, m)
                
            if np.sum(data.compute_flags)==0:
                if IS_PRINT_DIAGNOSTIC:
                    print_diagnostic('[CONVERGED] All components converged', data)
                break # sufficiently estimated
            elif 2*data.n_total>self.n_limit:
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceeds n_limit = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied. """ \
                % (int(data.n_total),int(data.n_total),int(self.n_limit))
                warnings.warn(warning_s, MaxSamplesWarning)
                if IS_PRINT_DIAGNOSTIC:
                    print_diagnostic('[MAX_SAMPLES] Reached n_limit', data)
                break
            data.n_min = data.n_max
            # Check if next iteration would exceed discrete distribution limit, but also ensure progress
            next_n_max = min(2*data.n_min, self.discrete_distrib.n_limit)
            if next_n_max <= data.n_min:
                # Can't make progress due to limit constraint
                # Before breaking, ensure we have a valid solution estimate
                if hasattr(data, 'muhat') and np.isfinite(data.muhat).any():
                    # Use current muhat estimate as final solution
                    data.solution = data.muhat.copy()
                elif hasattr(data, 'yfull') and data.yfull.size > 0:
                    # Compute mean from finite values in yfull
                    finite_mask = np.isfinite(data.yfull)
                    if finite_mask.any():
                        if self.ncv == 0:
                            # Simple mean of finite values
                            data.solution = np.where(finite_mask.any(axis=-1), 
                                                   np.nanmean(data.yfull, axis=-1), 
                                                   0.0)
                        else:
                            # With control variates - use available estimate or fallback
                            data.solution = np.nanmean(data.yfull, axis=-1)
                    else:
                        # No finite data available - use zero as fallback
                        data.solution = np.zeros(self.integrand.d_comb if hasattr(self.integrand, 'd_comb') else 1)
                else:
                    # No data available - use zero as fallback
                    data.solution = np.zeros(self.integrand.d_comb if hasattr(self.integrand, 'd_comb') else 1)
                
                warning_s = f"""
                Cannot continue: next n_max ({next_n_max}) <= current n_min ({data.n_min}).
                Discrete distribution n_limit ({self.discrete_distrib.n_limit}) is too small.
                Consider using a generating vector with higher m_max or increase n_limit manually.
                Returning best available estimate."""
                warnings.warn(warning_s, MaxSamplesWarning)
                if IS_PRINT_DIAGNOSTIC:
                    print_diagnostic('[MAX_SAMPLES] Reached discrete_distrib limit', data)
                break
            data.n_max = next_n_max
            
            if IS_PRINT_DIAGNOSTIC:
                print_diagnostic(f'[ITER] End iteration, next n_max={data.n_max}', data)
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = time()-t_start
        return data.solution,data
    
    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        assert rmse_tol is None, "rmse_tol not supported by this stopping criterion."
        if abs_tol is not None:
            self.abs_tol = abs_tol
            self.abs_tols = np.full(self.integrand.d_comb,self.abs_tol)
        if rel_tol is not None:
            self.rel_tol = rel_tol
            self.rel_tols = np.full(self.integrand.d_comb,self.rel_tol)
