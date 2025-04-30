from ._true_measure import TrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from numpy.linalg import cholesky, slogdet
from scipy.stats import norm, multivariate_normal
from scipy.linalg import eigh


class Gaussian(TrueMeasure):
    """
    Normal Measure.
    
    >>> g = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2],covariance=[[9,4],[4,5]])
    >>> g.gen_samples(4)
    array([[-3.73644286, -0.39366932],
           [ 6.25616165,  4.91776605],
           [ 0.30243592, -0.85231329],
           [ 1.71422011,  5.0055737 ]])
    >>> g
    Gaussian (TrueMeasure Object)
        mean            [1 2]
        covariance      [[9 4]
                        [4 5]]
        decomp_type     PCA
    """

    def __init__(self, sampler, mean=0., covariance=1., decomp_type='PCA'):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            mean (float): mu for Normal(mu,sigma^2)
            covariance (np.ndarray): sigma^2 for Normal(mu,sigma^2). 
                A float or d (dimension) vector input will be extended to covariance*np.eye(d)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        self.parameters = ['mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.domain = np.array([[0,1]])
        self._parse_sampler(sampler)
        self._parse_gaussian_params(mean,covariance,decomp_type)
        self.range = np.array([[-np.inf,np.inf]])
        super(Gaussian,self).__init__()
    
    def _parse_gaussian_params(self, mean, covariance, decomp_type):
        self.decomp_type = decomp_type.upper()
        self.mean = mean
        self.covariance = covariance
        if np.isscalar(mean):
            mean = np.tile(mean,self.d)
        if np.isscalar(covariance):
            covariance = covariance*np.eye(self.d)
        self.mu = np.array(mean)
        self.sigma = np.array(covariance)
        if self.sigma.shape==(self.d,):
            self.sigma = np.diag(self.sigma)
        self.sigma = (self.sigma+self.sigma.T)/2
        if not (len(self.mu)==self.d and self.sigma.shape==(self.d,self.d)):
            raise DimensionError('''
                    mean must have length d and
                    covariance must be of shape d x d''')
        if self.decomp_type == 'PCA':
            evals,evecs = eigh(self.sigma) # get eigenvectors and eigenvalues for
            evecs = evecs*(1-2*(evecs[0]<0)) # force first entries of eigenvectors to be positive
            order = np.argsort(-evals)
            self.a = np.dot(evecs[:,order],np.diag(np.sqrt(evals[order])))
        elif self.decomp_type == 'CHOLESKY':
            self.a = cholesky(self.sigma) #Fred changed this
        else:
            raise ParameterError("decomp_type should be 'PCA' or 'Cholesky'") 
        self.mvn_scipy = multivariate_normal(mean=self.mu,cov=self.sigma, allow_singular=True)

    def _transform(self, x):
        return self.mu + norm.ppf(x)@self.a.T

    def _weight(self, t):
        return self.mvn_scipy.pdf(t)
    
    def _spawn(self, sampler, dimension):
        if dimension==self.d: # don't do anything if the dimension doesn't change
            spawn = Gaussian(sampler,mean=self.mu,covariance=self.covariance,decomp_type=self.decomp_type)
        else:
            m = self.mu[0]
            c = self.sigma[0,0]
            expected_cov = c*np.eye(int(self.d))
            if not ( (self.mu==m).all() and (self.sigma==expected_cov).all() ):
                raise DimensionError('''
                        In order to spawn a Gaussian measure
                        mean (mu) must be all the same and 
                        covariance must be a scaler times I''')
            spawn = Gaussian(sampler,mean=m,covariance=c,decomp_type=self.decomp_type)
        return spawn

class Normal(Gaussian): pass
