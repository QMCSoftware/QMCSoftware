from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian, Lebesgue, Uniform
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from ..true_measure.abstract_true_measure import AbstractTrueMeasure
from ..util import ParameterError
import numpy as np

class BayesianLRCoeffs(Integrand):
    """
    Logistic Regression Coefficients computed as the posterior mean in a Bayesian framework.
    
    >>> blrcoeffs = BayesianLRCoeffs(DigitalNetB2(3,seed=7),feature_array=np.arange(8).reshape((4,2)),response_vector=[0,0,1,1])
    >>> x = blrcoeffs.discrete_distrib.gen_samples(2**10)
    >>> y = blrcoeffs.f(x)
    >>> y.shape
    (1024, 2, 3)
    >>> y.mean(0)
    array([[ 0.04948003, -0.01238017, -0.07146091],
           [ 0.02366789,  0.02366789,  0.02366789]])
    """

    def __init__(self, sampler, feature_array, response_vector, prior_mean=0, prior_covariance=10):
        """
        Args:
            sampler (AbstractDiscreteDistribution/AbstractTrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            feature_array (np.ndarray): N samples by d-1 dimensions array of input features
            response_vector (np.ndarray): length N array of binary responses corresponding to each
            prior_mean (np.ndarray): length d array of prior means, one for each coefficient. 
                The first d-1 inputs correspond to the d-1 features. 
                The last input corresponds to the intercept coefficient.
            prior_covariance (np.ndarray): d x d prior covariance array whose indexing is consistent with the prior mean.
        """
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.sampler = sampler
        self.true_measure = Gaussian(self.sampler, mean=self.prior_mean, covariance=self.prior_covariance)
        self.feature_array = np.array(feature_array,dtype=float)
        self.response_vector = np.array(response_vector,dtype=float)
        obs,dm1 = self.feature_array.shape
        self.num_coeffs = dm1+1
        if self.num_coeffs!=self.true_measure.d:
            ParameterError("sampler must have dimension one more than the number of features in the feature_array.")
        if self.response_vector.shape!=(obs,) or ((self.response_vector!=0)&(self.response_vector!=1)).any():
            ParameterError("response_vector must have the same length as feature_array and contain only 0 or 1 entries.")
        self.feature_array = np.column_stack((self.feature_array,np.ones((obs,1))))
        super(BayesianLRCoeffs,self).__init__(dimension_indv=(2,self.num_coeffs),dimension_comb=self.num_coeffs,parallel=False)
        
    def g(self, x, compute_flags):
        z = x@self.feature_array.T
        z1 = z*self.response_vector
        with np.errstate(over='ignore'):
            den = np.exp(np.sum(z1-np.log(1+np.exp(z)),1))[:,None]
        y = np.zeros((len(x),2,self.num_coeffs),dtype=float)
        y[:,0] = x*den
        y[:,1] = den
        return y
    
    def _spawn(self, level, sampler):
        return BayesianLRCoeffs(
            sampler = sampler,
            feature_array = self.feature_array,
            response_vector = self.response_vector,
            prior_mean = self.prior_mean,
            prior_covariance = self.prior_covariance)
    
    def bound_fun(self, bound_low, bound_high):
        num_bounds_low,den_bounds_low = bound_low[0],bound_low[1]
        num_bounds_high,den_bounds_high = bound_high[0],bound_high[1]
        comb_bounds_low = minimum.reduce([num_bounds_low/den_bounds_low,num_bounds_high/den_bounds_low,num_bounds_low/den_bounds_high,num_bounds_high/den_bounds_high])
        comb_bounds_high = maximum.reduce([num_bounds_low/den_bounds_low,num_bounds_high/den_bounds_low,num_bounds_low/den_bounds_high,num_bounds_high/den_bounds_high])
        violated = (den_bounds_low<=0)*(0<=den_bounds_high)
        comb_bounds_low[violated],comb_bounds_high[violated] = -np.inf,np.inf
        return comb_bounds_low,comb_bounds_high
    
    def dependency(self, comb_flags):
        return np.vstack((comb_flags,comb_flags))
