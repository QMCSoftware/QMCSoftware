from ._accumulate_data import AccumulateData
from numpy import *
from numpy.linalg import lstsq

class MLQMCData(AccumulateData):
    """
    Accumulated data for quasi-random sequence calculations,
    and store multi-level, multi-replications mean, variance, and cost values. 
    See the stopping criterion that utilize this object for references.
    """

    def __init__(self, stopping_crit, integrand, true_measure, discrete_distrib, levels_init, levels_max, n_init, replications):
        """
        Initialize data instance

        Args:
            stopping_crit (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            true_measure (TrueMeasure): A TrueMeasure instance
            discrete_distrib (DiscreteDistribution): a DiscreteDistribution instance
            replications (int): number of replications on each level
        """
        self.parameters = ['solution','n_total','n_level','levels','mean_level','var_level','bias_estimate']
        self.stopping_crit = stopping_crit
        self.integrand = integrand
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        # Set Attributes
        self.levels = int(levels_init + 1)
        self.n_level = zeros(self.levels)
        self.eval_level = tile(True,self.levels)
        self.replications = replications
        self.n_init = n_init
        self.mean_level_reps = tile(0.,(self.levels,int(self.replications)))
        self.mean_level = tile(0.,self.levels)
        self.var_level = tile(inf,self.levels)
        self.cost_level = tile(0.,self.levels)
        self.var_cost_ratio_level = tile(inf,self.levels)
        self.bias_estimate = inf
        self.solution = None
        self.n_total = 0
        self.level_integrands = []
        super(MLQMCData,self).__init__()

    def update_data(self):
        """ See abstract method. """
        # update sample sums
        for l in range(self.levels):
            if not self.eval_level[l]:
                # nothing to do on this level
                continue
            if l==len(self.level_integrands):
                # haven't spawned this level's integrand yet
                self.level_integrands += [self.integrand.spawn(levels=tile(l,int(self.replications)))]
            # reset dimension
            n_max = self.n_init if self.n_level[l]==0 else 2*self.n_level[l]
            for r in range(int(self.replications)):
                integrand_l = self.level_integrands[l][r]
                samples = integrand_l.discrete_distrib.gen_samples(n_min=self.n_level[l],n_max=n_max)
                integrand_l.f(samples).squeeze()
                prev_sum = self.mean_level_reps[l,r]*self.n_level[l]
                self.mean_level_reps[l,r] = (integrand_l.sums[0]+prev_sum)/float(n_max)
                self.cost_level[l] = self.cost_level[l] + integrand_l.cost
            self.n_level[l] = n_max
            self.mean_level[l] = self.mean_level_reps[l].mean()
            self.var_level[l] = self.mean_level_reps[l].var()
            cost_per_sample = self.cost_level[l]/self.n_level[l]/self.replications
            self.var_cost_ratio_level[l] = self.var_level[l]/cost_per_sample # Required in cub_qmc_ml.py
        self._update_bias_estimate()
        self.n_total = self.replications*self.n_level.sum()
        self.solution = self.mean_level.sum()
        self.eval_level[:] = False # Reset active levels

    def _update_bias_estimate(self):
        A = ones((2,2))
        A[:,0] = range(self.levels-2, self.levels)
        y = ones(2)
        y[0] = log2(abs(self.mean_level_reps[self.levels-2].mean()))
        y[1] = log2(abs(self.mean_level_reps[self.levels-1].mean()))
        x = lstsq(A, y, rcond=None)[0]
        alpha = maximum(.5,-x[0])
        self.bias_estimate = 2**(x[1]+self.levels*x[0]) / (2**alpha - 1)
    
    def _add_level(self):
        """ Add another level to relevant attributes. """
        self.levels += 1
        if self.levels > len(self.n_level):
            self.n_level = hstack((self.n_level,0))
            self.eval_level = hstack((self.eval_level,True))
            self.mean_level_reps = vstack((self.mean_level_reps,zeros(int(self.replications))))
            self.mean_level = hstack((self.mean_level,0))
            self.var_level = hstack((self.var_level,inf))
            self.cost_level = hstack((self.cost_level,0))
            self.var_cost_ratio_level = hstack((self.var_cost_ratio_level,inf))

    def _add_level_MLMC(self):
        """ Add another level to relevant attributes. """
        self.levels += 1
        if not len(self.n_level) > self.levels:
            self.mean_level = hstack((self.mean_level, self.mean_level[-1] / 2**self.alpha))
            self.var_level = hstack((self.var_level, self.var_level[-1] / 2**self.beta))
            self.cost_per_sample = hstack((self.cost_per_sample, self.cost_per_sample[-1] * 2**self.gamma))
            self.n_level = hstack((self.n_level, 0.))
            self.sum_level = hstack((self.sum_level,zeros((2,1))))
            self.cost_level = hstack((self.cost_level, 0.))
        else:
            self.mean_level = absolute(self.sum_level[0,:self.levels+1]/self.n_level[:self.levels+1])
            self.var_level = maximum(0,self.sum_level[1,:self.levels+1]/self.n_level[:self.levels+1] - self.mean_level**2)
            self.cost_per_sample = self.cost_level[:self.levels+1]/self.n_level[:self.levels+1]