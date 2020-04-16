""" Definition of Gaussian, a concrete implementation of TrueMeasure """

from ._true_measure import TrueMeasure
from ..util import TransformError
from numpy import array, sqrt, eye, dot, pi, exp,dot
from numpy.linalg import cholesky, det, inv
from scipy.stats import norm


class Gaussian(TrueMeasure):
    """ Gaussian (Normal) TrueMeasure """

    parameters = ['mu', 'covariance']

    def __init__(self, distribution, mean=0, covariance=1):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            mean (float): mu for Normal(mu,sigma^2)
            covariance (float/ndarray): sigma^2 for Normal(mu,sigma^2)
                Note: an ndarray should be of shape dimension x dimension
                      a float value is equivalent to float_val*eye(dimension)
        """
        self.distribution = distribution
        self.mu = array(mean)
        self.covariance = array(covariance)
        self.d = distribution.dimension
        cov_d = self.covariance if self.covariance.shape==(self.d,self.d) else self.covariance*eye(self.d)
        self.sigma = cholesky(cov_d)
        super().__init__()
    
    def pdf(self, x):
        """
        Gaussian pdf
        Args:
            x (ndarray): dx1 matrix of samples from domain
        """
        x = x.reshape(self.d,1)
        mu = self.mu.reshape(self.d,1)
        density = (2*pi)**(-self.d/2) * det(self.sigma)**(-1/2) * \
            exp(-1/2 *  dot( dot((x-mu).T,inv(self.sigma)), x-mu) )
        return density

    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear Gaussian
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
        """
        if self.distribution.mimics == 'StdGaussian':
            # shift and stretch
            mimic_samples = self.mu + dot(samples,self.sigma)
        elif self.distribution.mimics == "StdUniform":
            # inverse CDF then shift and stretch
            mimic_samples = self.mu + dot(norm.ppf(samples),self.sigma)
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Gaussian'%self.distribution.mimics)
        return mimic_samples

    def transform_g_to_f(self, g):
        """
        Transform g, the origianl integrand, to f,
        the integrand accepting standard distribution samples. 
        
        Args:
            g (method): original integrand
        
        Returns:
            f (method): transformed integrand
        """
        f = lambda samples: g(self._tf_to_mimic_samples(samples))
        return f
    
    def gen_mimic_samples(self, *args, **kwargs):
        """
        Generate samples from the DiscreteDistribution object
        and transform them to mimic TrueMeasure samples
        
        Args:
            *args (tuple): Ordered arguments to self.distribution.gen_samples
            **kwrags (dict): Keyword arguments to self.distribution.gen_samples
        
        Return:
             mimic_samples (ndarray): samples from the DiscreteDistribution transformed to appear 
                                  to appear like the TrueMeasure object
        """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples
