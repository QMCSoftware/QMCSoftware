from ._true_measure import TrueMeasure
from ..util import TransformError,DimensionError, ParameterError
from ..discrete_distribution import Sobol
from numpy import *
from numpy.linalg import cholesky, det, inv, eigh
from scipy.stats import norm
from scipy.special import erfcinv

class Gaussian(TrueMeasure):
    """
    Normal Measure.
    
    >>> d = 2
    >>> s = Sobol(d,seed=7)
    >>> g = Gaussian(d,mean=1,covariance=1./4)
    >>> g
    Gaussian (TrueMeasure Object)
        d               2^(1)
        mean            1
        covariance      2^(-2)
        decomp_type     pca
    >>> g.transform(s.gen_samples(n_min=2,n_max=4))
    array([[1.68 , 1.08 ],
           [0.425, 0.72 ]])
    >>> g.weight(array([[.1,.2],[.3,.4]]))
    array([0.035, 0.116])
    >>> g.jacobian(array([[.25,.75],[.35,.85]]))
    array([2.476, 2.895])
    >>> g.transformer == g
    True
    >>> g.set_transform(Gaussian(2,mean=[0,0],covariance=[[2,0],[0,2]]))
    >>> g.transformer.transform(s.gen_samples(n_min=2,n_max=4))
    array([[ 1.923,  0.226],
           [-1.626, -0.791]])
    >>> g.transformer.jacobian(array([[.25,.75],[.35,.85]]))
    array([19.806, 23.158])
    >>> d_new = 4
    >>> s.set_dimension(d_new)
    >>> g.set_dimension(d_new)
    >>> g.transform(s.gen_samples(n_min=2,n_max=4))
    array([[1.68 , 1.08 , 0.971, 1.274],
           [0.425, 0.72 , 1.948, 0.801]])
    >>> Gaussian(2,mean=[1,2],covariance=[[1,.5],[.5,2]])
    Gaussian (TrueMeasure Object)
        d               2^(1)
        mean            [1 2]
        covariance      [[1.  0.5]
                        [0.5 2. ]]
        decomp_type     pca
    """

    parameters = ['d', 'mean', 'covariance', 'decomp_type']

    def __init__(self, dimension=1, mean=0., covariance=1., decomp_type='PCA'):
        """
        Args:
            dimension (int): dimension of the distribution
            mean (float): mu for Normal(mu,sigma^2)
            covariance (ndarray): sigma^2 for Normal(mu,sigma^2). 
                A float or d (dimension) vector input will be extended to covariance*eye(d)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        self.d = dimension
        self.decomp_type = decomp_type.lower()
        self._set_mean_cov(mean,covariance)
        self.transformer = self
        super(Gaussian,self).__init__()
    
    def _set_mean_cov(self, mean, covariance):
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
        if not (len(self.mu)==self.d and self.sigma.shape==(self.d,self.d)):
            raise DimensionError('''
                    mean must have length d and
                    covariance must be of shape d x d''')
        self._set_constants()
    
    def _set_constants(self):
        if self.decomp_type == 'pca':
            evals,evecs = eigh(self.sigma) # get eigenvectors and eigenvalues for
            order = argsort(-evals)
            self.a = dot(evecs[:,order],diag(sqrt(evals[order])))
        elif self.decomp_type == 'cholesky':
            self.a = cholesky(self.sigma).T
        self.det_sigma = det(self.sigma)
        self.det_a = sqrt(self.det_sigma)
        self.inv_sigma = inv(self.sigma)  
    
    def transform(self, x):
        return self.mu + norm.ppf(x)@self.a.T
    
    def jacobian(self, x):
        return self.det_a/norm.pdf(norm.ppf(x)).prod(1)

    def weight(self, x):
        const = (2*pi)**(-self.d/2) * self.det_sigma**(-1./2)
        delta = x-self.mu
        return const*exp(-((delta@self.inv_sigma)*delta).sum(1)/2)

    def set_dimension(self, dimension):
        m = self.mu[0]
        c = self.sigma[0,0]
        expected_cov = c*eye(int(self.d))
        if not ( (self.mu==m).all() and (self.sigma==expected_cov).all() ):
            raise DimensionError('''
                    In order to change dimension of Gaussian measure
                    mean (mu) must be all the same and 
                    covariance must be a scaler times I''')
        self.d = dimension
        self.mu = tile(m,int(self.d))
        self.sigma = c*eye(int(self.d))
        self._set_constants()
        if self.transformer!=self:
            self.transformer.set_dimension(self.d)
    