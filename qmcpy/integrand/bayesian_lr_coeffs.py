from ._integrand import Integrand
from ..discrete_distribution import DigitalNetB2
from ..true_measure import Gaussian, Lebesgue, Uniform
from ..discrete_distribution._discrete_distribution import DiscreteDistribution
from ..true_measure._true_measure import TrueMeasure
from ..util import ParameterError
from numpy import *

class BayesianLRCoeffs(Integrand):
    """
    Logistic Regression Coefficients computed as the posterior mean in a Bayesian framework.
    
    >>> blrcoeffs = BayesianLRCoeffs(DigitalNetB2(3,seed=7),feature_array=arange(8).reshape((4,2)),response_vector=[0,0,1,1])
    >>> x = blrcoeffs.discrete_distrib.gen_samples(2**10)
    >>> y = blrcoeffs.f(x)
    >>> y.shape
    (1024, 6)
    >>> y.mean(0)
    array([ 0.04639394, -0.01440543, -0.05498496,  0.02176581,  0.02176581,
            0.02176581])
    """

    def __init__(self, sampler, feature_array, response_vector, prior_mean=0, prior_covariance=10):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform
            feature_array (ndarray): N samples by d-1 dimensions array of input features
            response_vector (ndarray): length N array of binary responses corresponding to each
            prior_mean (ndarray): length d array of prior means, one for each coefficient. 
                The first d-1 inputs correspond to the d-1 features. 
                The last input coresponds to the intercept coefficient.
            prior_covariance (ndarray): d x d prior covariance array whose indexing is consistent with the prior mean.
        """
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.sampler = sampler
        self.true_measure = Gaussian(self.sampler, mean=self.prior_mean, covariance=self.prior_covariance)
        self.feature_array = array(feature_array,dtype=float)
        self.response_vector = array(response_vector,dtype=float)
        obs,dm1 = self.feature_array.shape
        self.num_coeffs = dm1+1
        if self.num_coeffs!=self.true_measure.d:
            ParameterError("sampler must have dimension one more than the number of features in the feature_array.")
        if self.response_vector.shape!=(obs,) or ((self.response_vector!=0)&(self.response_vector!=1)).any():
            ParameterError("response_vector must have the same length as feature_array and contain only 0 or 1 enteries.")
        self.feature_array = column_stack((self.feature_array,ones((obs,1))))
        self.dprime = 2*self.num_coeffs
        super(BayesianLRCoeffs,self).__init__(dprime=self.dprime,parallel=False)
        
    def g(self, x, compute_flags):
        z = x@self.feature_array.T
        z1 = z*self.response_vector
        with errstate(over='ignore'):
            den = exp(sum(z1-log(1+exp(z)),1))[:,None]
        y = zeros((len(x),2*self.num_coeffs),dtype=float)
        y[:,:self.num_coeffs] = x*den
        y[:,self.num_coeffs:] = den
        return y
    
    def _spawn(self, level, sampler):
        return BayesianLRCoeffs(
            sampler = sampler,
            feature_array = self.feature_array,
            response_vector = self.response_vector,
            prior_mean = self.prior_mean,
            prior_covariance = self.prior_covariance)
    
    def bound_fun(self, bound_low, bound_high):
        num_bounds_low,den_bounds_low = bound_low[:self.num_coeffs],bound_low[self.num_coeffs:]
        num_bounds_high,den_bounds_high = bound_high[:self.num_coeffs],bound_high[self.num_coeffs:]
        comb_bounds_low = minimum.reduce([num_bounds_low/den_bounds_low,num_bounds_high/den_bounds_low,num_bounds_low/den_bounds_high,num_bounds_high/den_bounds_high])
        comb_bounds_high = maximum.reduce([num_bounds_low/den_bounds_low,num_bounds_high/den_bounds_low,num_bounds_low/den_bounds_high,num_bounds_high/den_bounds_high])
        violated = (den_bounds_low<=0)*(0<=den_bounds_high)
        comb_bounds_low[violated],comb_bounds_high[violated] = -inf,inf
        return comb_bounds_low,comb_bounds_high
    
    def dependency(self, flags_comb):
        return hstack((flags_comb,flags_comb))