from ._true_measure import TrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from numpy import *
from numpy.linalg import cholesky, slogdet
from scipy.stats import norm, multivariate_normal
from scipy.linalg import eigh


class Gaussian(TrueMeasure):
    """
    Normal Measure.
    
    >>> g = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2],covariance=[[9,4],[4,5]])
    >>> g.gen_samples(4)
    array([[-4.40566397,  1.31271715],
           [ 4.12464307,  2.70056246],
           [ 0.85301528, -0.88218749],
           [ 0.99611301,  3.16934534]])
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
            covariance (ndarray): sigma^2 for Normal(mu,sigma^2). 
                A float or d (dimension) vector input will be extended to covariance*eye(d)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        self.parameters = ['mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.domain = array([[0,1]])
        self._parse_sampler(sampler)
        self._parse_gaussian_params(mean,covariance,decomp_type)
        self.range = array([[-inf,inf]])
        super(Gaussian,self).__init__()
    
    def _parse_gaussian_params(self, mean, covariance, decomp_type):
        self.decomp_type = decomp_type.upper()
        self.mean = mean
        self.covariance = covariance
        if isscalar(mean):
            mean = tile(mean,self.d)
        if isscalar(covariance):
            covariance = covariance*eye(self.d)
        self.mu = array(mean)
        self.sigma = array(covariance)
        if self.sigma.shape==(self.d,):
            self.sigma = diag(self.sigma)
        self.sigma = (self.sigma+self.sigma.T)/2
        if not (len(self.mu)==self.d and self.sigma.shape==(self.d,self.d)):
            raise DimensionError('''
                    mean must have length d and
                    covariance must be of shape d x d''')
        if self.decomp_type == 'PCA':
            evals,evecs = eigh(self.sigma) # get eigenvectors and eigenvalues for
            evecs = evecs*(1-2*(evecs[0]<0)) # force first entries of eigenvectors to be positive
            order = argsort(-evals)
            self.a = dot(evecs[:,order],diag(sqrt(evals[order])))
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
            expected_cov = c*eye(int(self.d))
            if not ( (self.mu==m).all() and (self.sigma==expected_cov).all() ):
                raise DimensionError('''
                        In order to spawn a Gaussian measure
                        mean (mu) must be all the same and 
                        covariance must be a scaler times I''')
            spawn = Gaussian(sampler,mean=m,covariance=c,decomp_type=self.decomp_type)
        return spawn

class Normal(Gaussian): pass
