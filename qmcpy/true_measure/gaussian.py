from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
import numpy as np
from numpy.linalg import cholesky, slogdet
from scipy.stats import norm, multivariate_normal
from scipy.linalg import eigh
from typing import Union


class Gaussian(AbstractTrueMeasure):
    """
    Gaussian (Normal) distribution as described in [https://en.wikipedia.org/wiki/Multivariate_normal_distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution).
    
    Note:
        - `Normal` is an alias for `Gaussian`
    
    Examples:
        >>> true_measure = Gaussian(DigitalNetB2(2,seed=7),mean=[1,2],covariance=[[9,4],[4,5]])
        >>> true_measure(4)
        array([[ 4.40778501,  3.00772805],
               [-3.80150101,  1.58605376],
               [ 1.24089995,  3.27516695],
               [ 0.96180008, -0.52004296]])
        >>> true_measure
        Gaussian (AbstractTrueMeasure)
            mean            [1 2]
            covariance      [[9 4]
                             [4 5]]
            decomp_type     PCA
        
        With independent replications 

        >>> x = Gaussian(DigitalNetB2(3,seed=7,replications=2),mean=0,covariance=3)(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[-0.10044224,  2.09348242,  1.95102833],
                [ 0.96972313, -0.82447696, -0.7978428 ],
                [-1.53278114, -2.65447116, -2.26050713],
                [ 3.79928748,  0.62129823,  0.35269417]],
        <BLANKLINE>
               [[-1.18721904, -1.57108272,  1.15371635],
                [ 1.82012781,  0.12833591, -1.73542117],
                [-0.09769476,  2.30445062, -0.32093579],
                [ 0.54010329, -1.13296499,  3.44284423]]])
    """

    def __init__(self, sampler, mean=0., covariance=1., decomp_type='PCA'):
        """
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            mean (Union[float,np.ndarray]): Mean vector. 
            covariance (Union[float,np.ndarray]): Covariance matrix. A float or vector will be expanded into a diagonal matrix.  
            decomp_type (str): Method for decomposition for covariance matrix. Options include
             
                - `'PCA'` for principal component analysis, or 
                - `'Cholesky'` for cholesky decomposition.
        """
        self.parameters = ['mean', 'covariance', 'decomp_type']
        # default to transform from standard uniform
        self.domain = np.array([[0,1]])
        self._parse_sampler(sampler)
        self._parse_gaussian_params(mean,covariance,decomp_type)
        self.range = np.array([[-np.inf,np.inf]])
        super(Gaussian,self).__init__()
        assert self.mu.shape==(self.d,) and self.a.shape==(self.d,self.d)
    
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
            self.a = cholesky(self.sigma)
        else:
            raise ParameterError("decomp_type should be 'PCA' or 'Cholesky'") 
        self.mvn_scipy = multivariate_normal(mean=self.mu,cov=self.sigma, allow_singular=True)

    def _transform(self, x):
        return self.mu+np.einsum("...ij,kj->...ik",norm.ppf(x),self.a)

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
