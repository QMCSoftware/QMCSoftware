from ._discrete_distribution import DiscreteDistribution
from ..util import TransformError
from numpy import inf, zeros, random, array, apply_along_axis


class AcceptanceRejectionSampling(DiscreteDistribution):
    """
    Define
        - m(x) is pdf of measure we do not know how to generate from (mystery)
        - k(x) is pdf of measure we can generate discrete distribution samples from (known)
    
    Prodecure 
        1. samples s_i from k(x)
        2. samples u_i from Uniform(0,1)
        3. if u_i <= m(s_i)/(c*k(s_i)) ==> keep s_i
    
    Note: 
        this algorithm conservitively estimates c by taking 256 samples
        and approximating c ~= inflate_c_factor*max(m(s_i)/k(s_i) for i=1,...512)
    """

    parameters = ['c']

    def __init__(self, objective_pdf, measure_to_sample_from, draws_multiple=inf, inflate_c_factor=1):
        """
        Args:
            objective_pdf (function): pdf function of objective measure
            measure_to_sample_from (TrueMeasure): true measure we can sample from
            draws_multiple (float): will raise exception if drawing over n*draws_multiple samples
                                    when trying to get n samples
            inflate_c_factor (float): c = possibly inflate c to avoid underestimating
        """
        self.mimics = 'None'
        self.m = objective_pdf
        self.draws_multiple = draws_multiple
        self.measure = measure_to_sample_from
        self.distribution = self.measure.distribution
        self.dimension = self.distribution.dimension
        if not hasattr(self.measure,'pdf'):
            raise TransformError('measure_to_sample_from must have pdf method')
        self.k = self.measure.pdf
        if not ('IID' in type(self.distribution).__name__):
            raise TransformError('Acceptance rejection sampling only works with IID distributions.'+\
                                 'Make sure measure_to_samples_from has an IID distribution')
        # approximate c
        s = self.measure.gen_mimic_samples(256)
        md = apply_along_axis(self.m,1,s).squeeze()
        kd = apply_along_axis(self.k,1,s).squeeze()
        self.c = max( (md/kd) )
        super().__init__()

    def gen_samples(self, n):
        """
        Generate samples 

        Args:
            n (int): Number of observations to generate

        Returns:
            ndarray: n x d (dimension) array of samples
        """
        samples = array([sample for sample,keep in self._sample_generator(n) if keep])
        return samples

    def _sample_generator(self,n):
        self.successful_draws = 0 # successful draws
        self.total_draws = 0 # total draws
        max_draws = self.draws_multiple*n
        while self.successful_draws < n:
            keep = False
            candidate = self.measure.gen_mimic_samples(1).squeeze()
            md = self.m(candidate) # density at objective measure
            kd = self.k(candidate) # density at measure we sampled from
            u = random.rand(1)
            if u<= md/(self.c*kd):
                keep = True
                self.successful_draws += 1
            self.total_draws += 1
            if self.total_draws >= max_draws: 
                raise Exception('Drawn max samples of %d. Found %d successes, less than goal of %d.'%\
                                (self.total_draws,self.successful_draws,n))
            yield candidate,keep