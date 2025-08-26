from .abstract_stopping_criterion import AbstractStoppingCriterion
import numpy as np
from scipy.stats import norm

class AbstractCubMLMC(AbstractStoppingCriterion):
    
    def _get_next_samples(self, data):
        ns = np.ceil( np.sqrt(data.var_level/data.cost_per_sample) * 
                np.sqrt(data.var_level*data.cost_per_sample).sum() / 
                ((1-self.theta)*self.rmse_tol**2) )
        return ns.astype(int)
    
    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        assert rel_tol is None, "rel_tol not supported by this stopping criterion."
        if rmse_tol != None:
            self.rmse_tol = float(rmse_tol)
        elif abs_tol != None:
            self.rmse_tol = (float(abs_tol) / norm.ppf(1-self.alpha/2.))
    
    def _update_data(self, data):
        for l in range(data.levels+1):
            if l==len(data.level_integrands):
                # haven't spawned this level's integrand yet
                data.level_integrands += self.integrand.spawn(levels=int(l))
            integrand_l = data.level_integrands[l]
            if data.diff_n_level[l] > 0:
                # evaluate integral at sampling points samples
                n = data.diff_n_level[l]
                samples = integrand_l.discrete_distrib(n=n)
                pc,pf = integrand_l.f(samples)
                dp = pf-pc
                data.n_level[l] = data.n_level[l] + data.diff_n_level[l]
                data.sum_level[0,l] = data.sum_level[0,l] + dp.sum()
                data.sum_level[1,l] = data.sum_level[1,l] + (dp**2).sum()
                data.cost_level[l] = data.cost_level[l] + integrand_l.cost*n
        # compute absolute average, variance and cost
        data.mean_level = np.absolute(data.sum_level[0,:data.levels+1]/data.n_level[:data.levels+1])
        data.var_level = np.maximum(0,data.sum_level[1,:data.levels+1]/data.n_level[:data.levels+1] - data.mean_level**2)
        data.cost_per_sample = data.cost_level[:data.levels+1]/data.n_level[:data.levels+1]
        # fix to cope with possible zero values for data.mean_level and data.var_level
        # (can happen in some applications when there are few samples)
        for l in range(2,data.levels+1):
            data.mean_level[l] = np.maximum(data.mean_level[l], .5*data.mean_level[l-1]/2**data.alpha)
            data.var_level[l] = np.maximum(data.var_level[l], .5*data.var_level[l-1]/2**data.beta)
        # use linear regression to estimate alpha, beta, gamma if not given
        a = np.ones((data.levels,2))
        a[:,0] = np.arange(1,data.levels+1)
        if self.alpha0 <= 0:
            x = np.linalg.lstsq(a,np.log2(data.mean_level[1:]),rcond=None)[0]
            data.alpha = np.maximum(.5,-x[0])
        if self.beta0 <= 0:
            x = np.linalg.lstsq(a,np.log2(data.var_level[1:]),rcond=None)[0]
            data.beta = np.maximum(.5,-x[0])
        if self.gamma0 <= 0:
            x = np.linalg.lstsq(a,np.log2(data.cost_per_sample[1:]),rcond=None)[0]
            data.gamma = np.maximum(.5,x[0])
        data.n_total = data.n_level.sum()

    def _add_level(self, data):
        data.levels += 1
        if not len(data.n_level) > data.levels:
            data.mean_level = np.hstack((data.mean_level, data.mean_level[-1] / 2**data.alpha))
            data.var_level = np.hstack((data.var_level, data.var_level[-1] / 2**data.beta))
            data.cost_per_sample = np.hstack((data.cost_per_sample, data.cost_per_sample[-1] * 2**data.gamma))
            data.n_level = np.hstack((data.n_level, 0))
            data.sum_level = np.hstack((data.sum_level,np.zeros((2,1))))
            data.cost_level = np.hstack((data.cost_level, 0.))
        else:
            data.mean_level = np.absolute(data.sum_level[0,:data.levels+1]/data.n_level[:data.levels+1])
            data.var_level = np.maximum(0,data.sum_level[1,:data.levels+1]/data.n_level[:data.levels+1] - data.mean_level**2)
            data.cost_per_sample = data.cost_level[:data.levels+1]/data.n_level[:data.levels+1]