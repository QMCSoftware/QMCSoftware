from ._accumulate_data import AccumulateData
from numpy import tile, zeros, random, inf, hstack, vstack

class MLQMCData(AccumulateData):
    """
    Accumulated data for quasi-random sequence calculations,
    and store multi-level, multi-replications mean, variance, and cost values. 

    Reference:
        M.B. Giles and B.J. Waterhouse. 'Multilevel quasi-Monte Carlo path simulation'.
        pp.165-181 in Advanced Financial Modelling, in Radon Series on Computational and Applied Mathematics,
        de Gruyter, 2009. http://people.maths.ox.ac.uk/~gilesm/files/radon.pdf
    """

    parameters = ['levels','n_level','mean_level','var_level','bias_estimate','n_total']

    def __init__(self, stopping_criterion, integrand, n_init, replications):
        """
        Initialize data instance

        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            replications (int): number of replications on each level
        """
        # Extract QMCPy objects
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        # Set Attributes
        self.levels = 2
        self.n_level = zeros(self.levels,dtype=int)
        self.eval_level = tile(True,self.levels)
        self.replications = replications
        self.n_init = n_init
        self.mean_level_reps = tile(0.,(self.levels,self.replications))
        self.mean_level = tile(0.,self.levels)
        self.var_level = tile(inf,self.levels)
        self.cost_level = tile(inf,self.levels)
        self.bias_estimate = inf
        self.solution = None
        self.n_total = 0
        # get seeds for each replication
        random.seed(self.distribution.seed)
        self.seeds = random.randint(0,1000000,(self.levels,self.replications))
        super().__init__()

    def update_data(self):
        """ See abstract method. """
        # update sample sums
        for l in range(self.levels):
            if not self.eval_level[l]:
                # nothing to do on this level
                continue
            # reset dimension
            new_dim = self.integrand.dim_at_level(l)
            self.measure.set_dimension(new_dim)
            n_max = self.n_init if self.n_level[l]==0 else 2*self.n_level[l]
            for r in range(self.replications):
                self.distribution.set_seed(self.seeds[l,r]) # reset seed
                samples = self.distribution.gen_samples(n_min=self.n_level[l],n_max=n_max)
                sums,cost = self.integrand.f(samples,l=l)
                prev_sum = self.mean_level_reps[l,r]*self.n_level[l]
                self.mean_level_reps[l,r] = (sums[0]+prev_sum)/float(n_max)
            self.n_level[l] = n_max
            self.mean_level[l] = self.mean_level_reps[l].mean()
            self.var_level[l] = self.mean_level_reps[l].var()
            self.cost_level[l] = self.var_level[l]/(new_dim*self.n_level[l])
        self.bias_estimate = max(.5*abs(self.mean_level[self.levels-2]),\
                                 abs(self.mean_level[self.levels-1]))
        self.n_total = self.replications*self.n_level.sum()
        self.solution = self.mean_level.sum()
    
    def add_level(self):
        """ Add another level to relevent attributes. """
        self.levels += 1
        self.n_level = hstack((self.n_level,0))
        self.eval_level = hstack((self.eval_level,True))
        self.mean_level_reps = vstack((self.mean_level_reps,zeros(self.replications)))
        self.mean_level = hstack((self.mean_level,0))
        self.var_level = hstack((self.var_level,inf))
        self.cost_level = hstack((self.cost_level,inf))
        self.seeds = vstack((self.seeds,random.randint(0,1000000,self.replications)))
