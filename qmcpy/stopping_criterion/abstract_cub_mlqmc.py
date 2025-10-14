from .abstract_stopping_criterion import AbstractStoppingCriterion
import numpy as np
from scipy.stats import norm


class AbstractCubMLQMC(AbstractStoppingCriterion):
    
    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        assert rel_tol is None, "rel_tol not supported by this stopping criterion."
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = (float(abs_tol) / norm.ppf(1-self.alpha/2.))
    
    def update_data(self, data):
        # update sample sums
        for l in range(data.levels):
            if not data.eval_level[l]:
                # nothing to do on this level
                continue
            if l==len(data.level_integrands):
                # haven't spawned this level's integrand yet
                data.level_integrands += self.integrand.spawn(levels=l)
            # reset dimension
            n_max = self.n_init if data.n_level[l]==0 else 2*data.n_level[l]
            integrand_l = data.level_integrands[l]
            n_min = data.n_level[l]
            samples = integrand_l.discrete_distrib(n_min=n_min,n_max=n_max)
            n = n_max-n_min
            pc,pf = integrand_l.f(samples).squeeze()
            dp = pf-pc
            prev_sum = data.mean_level_reps[l]*data.n_level[l]
            data.mean_level_reps[l] = (dp.sum(-1)+prev_sum)/float(n_max)
            data.cost_level[l] = data.cost_level[l] + self.replications*n*integrand_l.cost
            data.n_level[l] = n_max
            data.mean_level[l] = data.mean_level_reps[l].mean()
            data.var_level[l] = data.mean_level_reps[l].var()
            cost_per_sample = data.cost_level[l]/data.n_level[l]/self.replications
            data.var_cost_ratio_level[l] = data.var_level[l]/cost_per_sample
        self._update_bias_estimate(data)
        data.n_total = self.replications*data.n_level.sum()
        data.solution = data.mean_level.sum()
        data.eval_level[:] = False # Reset active levels

    def _update_bias_estimate(self, data):
        A = np.ones((2,2))
        A[:,0] = range(data.levels-2, data.levels)
        y = np.ones(2)
        y[0] = np.log2(abs(data.mean_level_reps[data.levels-2].mean()))
        y[1] = np.log2(abs(data.mean_level_reps[data.levels-1].mean()))
        x = np.linalg.lstsq(A,y,rcond=None)[0]
        alpha = max(.5,-x[0])
        data.bias_estimate = 2**(x[1]+data.levels*x[0]) / (2**alpha - 1)
    
    def _add_level(self, data):
        # Add another level to relevant attributes.
        data.levels += 1
        if data.levels > len(data.n_level):
            data.n_level = np.hstack((data.n_level,0))
            data.eval_level = np.hstack((data.eval_level,True))
            data.mean_level_reps = np.vstack((data.mean_level_reps,np.zeros(int(self.replications))))
            data.mean_level = np.hstack((data.mean_level,0))
            data.var_level = np.hstack((data.var_level,np.inf))
            data.cost_level = np.hstack((data.cost_level,0))
            data.var_cost_ratio_level = np.hstack((data.var_cost_ratio_level,np.inf))

    def _add_level_MLMC(self, data):
        # Add another level to relevant attributes.
        data.levels += 1
        if not len(data.n_level) > data.levels:
            data.mean_level = np.hstack((data.mean_level, data.mean_level[-1] / 2**data.alpha))
            data.var_level = np.hstack((data.var_level, data.var_level[-1] / 2**data.beta))
            data.cost_per_sample = np.hstack((data.cost_per_sample, data.cost_per_sample[-1] * 2**data.gamma))
            data.n_level = np.hstack((data.n_level, 0.))
            data.sum_level = np.hstack((data.sum_level,np.zeros((2,1))))
            data.cost_level = np.hstack((data.cost_level, 0.))
        else:
            data.mean_level = np.absolute(data.sum_level[0,:data.levels+1]/data.n_level[:data.levels+1])
            data.var_level = np.maximum(0,data.sum_level[1,:data.levels+1]/data.n_level[:data.levels+1] - data.mean_level**2)
            data.cost_per_sample = data.cost_level[:data.levels+1]/data.n_level[:data.levels+1]
