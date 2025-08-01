from ._accumulate_data import AccumulateData
from numpy import *
from numpy.linalg import lstsq


class MLMCData(AccumulateData):
    """
    Accumulated data for IIDDistribution calculations,
    and store multi-level mean, variance, and cost values.
    See the stopping criterion that utilize this object for references.
    """

    def __init__(self, stopping_crit, integrand, true_measure, discrete_distrib, levels_init, n_init, 
                alpha0, beta0, gamma0):
        """
        Initialize data instance

        Args:
            stopping_crit (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            true_measure (TrueMeasure): A TrueMeasure instance
            discrete_distrib (DiscreteDistribution): a DiscreteDistribution instance
            levels_init (int): initial number of levels
            n_init (int): initial number of samples per level
            alpha0 (float): weak error is O(2^{-alpha0*level})
            beta0 (float): variance is O(2^{-beta0*level})
            gamma0 (float): sample cost is O(2^{gamma0*level})
        """
        self.parameters = ['solution','n_total','levels','n_level','mean_level','var_level', 
            'cost_per_sample','alpha','beta','gamma']
        self.stopping_crit = stopping_crit
        self.integrand = integrand
        self.true_measure = true_measure
        self.discrete_distrib = discrete_distrib
        # Set Attributes
        self.levels = int(levels_init)
        self.n_level = zeros(self.levels+1)
        self.sum_level = zeros((2,self.levels+1))
        self.cost_level = zeros(self.levels+1)
        self.diff_n_level = tile(n_init,self.levels+1)
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = gamma0
        self.alpha = maximum(0,self.alpha0)
        self.beta = maximum(0,self.beta0)
        self.gamma = maximum(0,self.gamma0)
        self.solution = None
        self.n_total = 0
        self.time_integrate = 0
        self.level_integrands = []
        super(MLMCData,self).__init__()

    def update_data(self):
        """ See abstract method. """
        # update sample sums
        for l in range(self.levels+1):
            if l==len(self.level_integrands):
                # haven't spawned this level's integrand yet
                self.level_integrands += self.integrand.spawn(levels=int(l))
            integrand_l = self.level_integrands[l]
            if self.diff_n_level[l] > 0:
                # evaluate integral at sampling points samples
                samples = integrand_l.discrete_distrib.gen_samples(n=self.diff_n_level[l])
                integrand_l.f(samples).squeeze()
                self.n_level[l] = self.n_level[l] + self.diff_n_level[l]
                self.sum_level[0,l] = self.sum_level[0,l] + integrand_l.sums[0]
                self.sum_level[1,l] = self.sum_level[1,l] + integrand_l.sums[1]
                self.cost_level[l] = self.cost_level[l] + integrand_l.cost
        # compute absolute average, variance and cost
        self.mean_level = absolute(self.sum_level[0,:self.levels+1]/self.n_level[:self.levels+1])
        self.var_level = maximum(0,self.sum_level[1,:self.levels+1]/self.n_level[:self.levels+1] - self.mean_level**2)
        self.cost_per_sample = self.cost_level[:self.levels+1]/self.n_level[:self.levels+1]
        # fix to cope with possible zero values for self.mean_level and self.var_level
        # (can happen in some applications when there are few samples)
        for l in range(2,self.levels+1):
            self.mean_level[l] = maximum(self.mean_level[l], .5*self.mean_level[l-1]/2**self.alpha)
            self.var_level[l] = maximum(self.var_level[l], .5*self.var_level[l-1]/2**self.beta)
        # use linear regression to estimate alpha, beta, gamma if not given
        a = ones((self.levels,2))
        a[:,0] = arange(1,self.levels+1)
        if self.alpha0 <= 0:
            x = lstsq(a,log2(self.mean_level[1:]),rcond=None)[0]
            self.alpha = maximum(.5,-x[0])
        if self.beta0 <= 0:
            x = lstsq(a,log2(self.var_level[1:]),rcond=None)[0]
            self.beta = maximum(.5,-x[0])
        if self.gamma0 <= 0:
            x = lstsq(a,log2(self.cost_per_sample[1:]),rcond=None)[0]
            self.gamma = maximum(.5,x[0])
        self.n_total = self.n_level.sum()

    def _add_level(self):
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