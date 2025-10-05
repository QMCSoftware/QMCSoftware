from .abstract_stopping_criterion import AbstractStoppingCriterion, IS_PRINT_DIAGNOSTIC, print_diagnostic
from ..util.data import Data

from ..util import MaxSamplesWarning, ParameterError, ParameterWarning, CubatureWarning
from ..integrand import AbstractIntegrand
import numpy as np
from time import time
import warnings


class AbstractCubQMCLDG(AbstractStoppingCriterion):
    
    def __init__(self, integrand, abs_tol, rel_tol, n_init, n_max, fudge, check_cone,
        control_variates, control_variate_means, update_beta, ptransform,
        coefv, allowed_distribs, cast_complex, error_fun):
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
        # Handle error_fun conversion from string to function
        if isinstance(error_fun, str):
            if error_fun.upper() == "EITHER":
                error_fun = lambda sv, abs_tol, rel_tol: np.maximum(abs_tol, abs(sv) * rel_tol)
            elif error_fun.upper() == "BOTH":
                error_fun = lambda sv, abs_tol, rel_tol: np.minimum(abs_tol, abs(sv) * rel_tol)
            else:
                raise ParameterError("str error_fun must be 'EITHER' or 'BOTH'")
        self.error_fun = error_fun
        self.integrand = integrand
        self.true_measure = self.integrand.true_measure
        self.discrete_distrib = self.integrand.discrete_distrib
        self.d = self.discrete_distrib.d
        self.d_indv = self.integrand.d_indv
        self.cv = list(np.atleast_1d(control_variates))
        self.ncv = len(self.cv)
        self.cv_mu = np.array(control_variate_means) if self.ncv>0 else np.empty((self.ncv,)+self.d_indv)
        if self.ncv > 0:
            self.cv_mu = self.cv_mu if self.cv_mu.ndim>1 else self.cv_mu.reshape(self.ncv,-1)
        self.is_first_resume_iteration = False
        if self.cv_mu.shape!=((self.ncv,)+self.d_indv):
            raise ParameterError('''Control variate means should have shape (len(control variates),d_indv).''')
        for cv in self.cv:
            if (cv.discrete_distrib!=self.discrete_distrib) or (not isinstance(cv,AbstractIntegrand)) or (cv.d_indv!=self.d_indv):
                raise ParameterError('''
                        Each control variates discrete distribution must be an AbstractIntegrand instance 
                        with the same discrete distribution as the main integrand. d_indv must also match 
                        that of the main integrand instance for each control variate.''')
        self.update_beta = update_beta
        if self.ncv>0:
            self.parameters += ['cv','cv_mu','update_beta']
        super().__init__(allowed_distribs=allowed_distribs,allow_vectorized_integrals=True)
    
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
            self.data = resume
            if hasattr(self.data, 'n_total'):
                # Set n_min to n_total so next batch starts after previous samples
                self.data.n_min = int(self.data.n_total)
            self.is_first_resume_iteration = True
            if IS_PRINT_DIAGNOSTIC:
                print_diagnostic('[RESUME] Resuming from data', self.data)
        else:
            # Initialize from scratch
            self.data = Data(
                parameters = [
                    'solution',
                    'comb_bound_low',
                    'comb_bound_high',
                    'comb_flags',
                    'n_total',
                    'n',
                    'time_integrate'])
            self.data.flags_indv = np.tile(False,self.integrand.d_indv)
            self.data.compute_flags = np.tile(True,self.integrand.d_indv)
            self.data.m = np.tile(self.m_min,self.integrand.d_indv)
            self.data.n_min = 0
            self.data.indv_bound_low = np.tile(-np.inf,self.integrand.d_indv)
            self.data.indv_bound_high = np.tile(np.inf,self.integrand.d_indv)
            self.data.solution_indv = np.tile(np.nan,self.integrand.d_indv)
            self.data.solution = np.nan
            self.data.xfull = np.empty((0,self.integrand.d))
            self.data.yfull = np.empty((0,)+self.integrand.d_indv)
            self.is_first_resume_iteration = False
            
        while True:
            m = self.data.m.max()
            n_min = self.data.n_min
            n_max = int(2**m)
            n = int(n_max-n_min)
            
            if not self.is_first_resume_iteration:  # generate new samples
                xnext = self.discrete_distrib(n_min=n_min,n_max=n_max)
                ynext = self.integrand.f(xnext,periodization_transform=self.ptransform,compute_flags=self.data.compute_flags)
                
                # Simple data accumulation - compatible with scalar and vector outputs
                self.data.xfull = np.vstack((self.data.xfull, xnext))
                if len(self.integrand.d_indv) == 0:  # scalar case
                    self.data.yfull = np.concatenate((self.data.yfull, ynext), axis=0)
                else:  # vector case
                    self.data.yfull = np.vstack((self.data.yfull, ynext))
                
                # Update individual solutions with simple mean
                for j in np.ndindex(self.integrand.d_indv):
                    if self.data.flags_indv[j]: continue
                    # Use all accumulated data for this component
                    if self.data.yfull.size > 0:
                        if len(self.integrand.d_indv) == 0:  # scalar case
                            self.data.solution_indv[j] = np.mean(self.data.yfull)
                            std_err = np.std(self.data.yfull) / np.sqrt(len(self.data.yfull)) if len(self.data.yfull) > 1 else np.inf
                        else:  # vector case
                            values = self.data.yfull[(...,) + j]
                            self.data.solution_indv[j] = np.mean(values)
                            std_err = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else np.inf
                        
                        # Basic bound computation
                        margin = self.fudge(m) * std_err
                        self.data.indv_bound_low[j] = self.data.solution_indv[j] - margin
                        self.data.indv_bound_high[j] = self.data.solution_indv[j] + margin
            
            # Compute combined bounds and solution
            self.data.comb_bound_low, self.data.comb_bound_high = self.integrand.bound_fun(
                self.data.indv_bound_low, self.data.indv_bound_high)
            self.abs_tols, self.rel_tols = np.full_like(self.data.comb_bound_low, self.abs_tol), np.full_like(self.data.comb_bound_low, self.rel_tol)
            fidxs = np.isfinite(self.data.comb_bound_low) & np.isfinite(self.data.comb_bound_high)
            slow, shigh, abs_tols, rel_tols = self.data.comb_bound_low[fidxs], self.data.comb_bound_high[fidxs], self.abs_tols[fidxs], self.rel_tols[fidxs]
            
            self.data.solution = np.tile(np.nan, self.data.comb_bound_low.shape)
            self.data.solution[fidxs] = 1/2*(slow+shigh+self.error_fun(slow, abs_tols, rel_tols)-self.error_fun(shigh, abs_tols, rel_tols))
            
            self.data.comb_flags = np.tile(False, self.data.comb_bound_low.shape)
            self.data.comb_flags[fidxs] = (shigh-slow) < (self.error_fun(slow, abs_tols, rel_tols)+self.error_fun(shigh, abs_tols, rel_tols))
            
            self.data.flags_indv = self.integrand.dependency(self.data.comb_flags)
            self.data.compute_flags = ~self.data.flags_indv
            self.data.n = 2**self.data.m
            self.data.n_total = self.data.n.max()
            
            if IS_PRINT_DIAGNOSTIC: 
                print_diagnostic("INFO: In each iteration", self.data)
            
            if np.sum(self.data.compute_flags)==0:
                break # stopping criterion met
            elif 2*self.data.n_total>self.n_max:
                # doubling samples would go over n_limit
                warning_s = """
                Already generated %d samples.
                Trying to generate %d new samples would exceed n_max = %d.
                No more samples will be generated.
                Note that error tolerances may not be satisfied.""" \
                % (int(self.data.n_total),int(self.data.n_total),int(self.n_max))
                warnings.warn(warning_s, MaxSamplesWarning)
                break
            else:
                self.data.n_min = n_max
                self.data.m += self.data.compute_flags
                
            # Reset the resume flag after first iteration
            if self.is_first_resume_iteration:
                self.is_first_resume_iteration = False
        
        self.data.integrand = self.integrand
        self.data.true_measure = self.true_measure
        self.data.discrete_distrib = self.discrete_distrib
        self.data.stopping_crit = self
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
