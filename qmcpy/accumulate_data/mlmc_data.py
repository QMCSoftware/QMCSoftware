from ._accumulate_data import AccumulateData
from numpy import zeros, absolute, maximum, tile, array, log2, arange, ones
from numpy.linalg import lstsq


class MLMCData(AccumulateData):
    """
    Accumulated data for IIDDistribution calculations,
    and store multi-level mean, variance, and cost values.

    Reference:
        M.B. Giles. 'Multi-level Monte Carlo path simulation'. 
        Operations Research, 56(3):607-617, 2008.
        http://people.maths.ox.ac.uk/~gilesm/files/OPRE_2008.pdf.
    """

    parameters = ['levels','n_level','mean_level','var_level','cost_per_sample',
        'alpha','beta','gamma']

    def __init__(self, stopping_criterion, integrand, levels_init, n_init, alpha0, beta0, gamma0):
        """
        Initialize data instance

        Args:
            stopping_criterion (StoppingCriterion): a StoppingCriterion instance
            integrand (Integrand): an Integrand instance
            levels_init (int): initial number of levels
            n_init (int): initial number of samples per level
            alpha0 (float): weak error is O(2^{-alpha0*level})
            beta0 (float): variance is O(2^{-beta0*level})
            gamma0 (float): sample cost is O(2^{gamma0*level})
        """
        # Extract QMCPy objects
        self.stopping_criterion = stopping_criterion
        self.integrand = integrand
        self.measure = self.integrand.measure
        self.distribution = self.measure.distribution
        # Set Attributes
        self.levels = levels_init
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
        super().__init__()

    def update_data(self):
        """ See abstract method. """
        # update sample sums
        for l in range(self.levels+1):
            if self.diff_n_level[l] > 0:
                # reset dimension
                new_dim = self.integrand.dim_at_level(l)
                self.measure.set_dimension(new_dim)
                # evaluate integral at sampleing points samples
                samples = self.distribution.gen_samples(n=self.diff_n_level[l])
                sums,cost = self.integrand.f(samples,l=l)
                self.n_level[l] = self.n_level[l] + self.diff_n_level[l]
                self.sum_level[0,l] = self.sum_level[0,l] + sums[0]
                self.sum_level[1,l] = self.sum_level[1,l] + sums[1]
                self.cost_level[l] = self.cost_level[l] + cost
        # compute absolute average, variance and cost
        self.mean_level = absolute(self.sum_level[0,:]/self.n_level)
        self.var_level = maximum(0,self.sum_level[1,:]/self.n_level - self.mean_level**2)
        self.cost_per_sample = self.cost_level/self.n_level
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