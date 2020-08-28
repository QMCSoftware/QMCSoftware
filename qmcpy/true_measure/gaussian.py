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
    
    >>> dd = Sobol(2,seed=7)
    >>> g = Gaussian(dd,mean=1,covariance=1./4)
    >>> g
    Gaussian (TrueMeasure Object)
        mean            1
        covariance      2^(-2)
        decomp_type     pca
    >>> g.gen_samples(n_min=4,n_max=8)
    array([[ 1.533,  0.676],
           [ 0.817,  1.351],
           [ 1.136,  1.011],
           [ 0.379, -0.194]])
    >>> g.set_dimension(4)
    >>> g.gen_samples(n_min=2,n_max=4)
    array([[1.309, 0.445, 1.128, 0.813],
           [0.634, 1.171, 0.362, 1.528]])
    >>> g2 = Gaussian(Sobol(2),mean=[1,2],covariance=[[1,.5],[.5,2]])
    >>> g2
    Gaussian (TrueMeasure Object)
        mean            [1 2]
        covariance      [[1.  0.5]
                        [0.5 2. ]]
        decomp_type     pca
    >>> g2.pdf(array([0,0]))
    array([[0.038]])
    """

    parameters = ['mean', 'covariance', 'decomp_type']

    def __init__(self, distribution, mean=0., covariance=1., decomp_type='PCA'):
        """
        Args:
            distribution (DiscreteDistribution): DiscreteDistribution instance
            mean (float): mu for Normal(mu,sigma^2)
            covariance (ndarray): sigma^2 for Normal(mu,sigma^2). 
                A float or d (dimension) vector input will be extended to covariance*eye(d)
            decomp_type (str): method of decomposition either  
                "PCA" for principal component analysis or 
                "Cholesky" for cholesky decomposition.
        """
        self.distribution = distribution
        self.d = distribution.dimension
        self.mean = mean
        self.covariance = covariance
        self.decomp_type = decomp_type.lower()
        if isscalar(mean):
            mean = tile(mean,self.d)
        if isscalar(covariance):
            covariance = covariance*eye(self.d)
        self.mu = array(mean)
        self.sigma = array(covariance)
        if self.sigma.shape==(self.d,):
            self.sigma = diag(self.sigma)
        if not (len(self.mu)==self.d and self.sigma.shape==(self.d,self.d)):
            raise DimensionError("mean must have length dimension and "+\
                "covariance must be of shapre dimension x dimension")
        self._assemble()
        super(Gaussian,self).__init__()
    
    def _assemble(self):
        """ Assemble decomposition of sigma. """
        if self.decomp_type == 'pca':
            evals,evecs = eigh(self.sigma) # get eigenvectors and eigenvalues for
            order = argsort(-evals)
            self.a = dot(evecs[:,order],diag(sqrt(evals[order]))).T
        elif self.decomp_type == 'cholesky':
            self.a = cholesky(self.sigma)
    
    def pdf(self, x):
        """ See abstract method. """
        x = x.reshape(self.d,1)
        mu = self.mu.reshape(self.d,1)
        density = (2*pi)**(-self.d/2.) * det(self.sigma)**(-1./2) * \
            exp(-1./2 *  dot( dot((x-mu).T,inv(self.sigma)), x-mu) )
        return density

    def _tf_to_mimic_samples(self, samples):
        """
        Transform samples to appear Gaussian
        
        Args:
            samples (ndarray): samples from a discrete distribution
        
        Return:
            ndarray: samples from the DiscreteDistribution transformed to mimic Gaussian.
        """
        if self.distribution.mimics == 'StdGaussian':
            std_gaussian_samples = samples
            mimic_samples = self.mu + dot(std_gaussian_samples, self.a)
        elif self.distribution.mimics == "StdUniform":
            std_gaussian_samples = norm.ppf(samples)
            mimic_samples = self.mu + dot(std_gaussian_samples, self.a)
            # mimic_samples = erfcinv(samples)
        else:
            raise TransformError(\
                'Cannot transform samples mimicing %s to Gaussian'%self.distribution.mimics)
        return mimic_samples

    def _transform_g_to_f(self, g):
        """ See abstract method. """
        def f(samples, *args, **kwargs):
            z = self._tf_to_mimic_samples(samples)
            y = g(z, *args, **kwargs)
            return y
        return f
    
    def gen_samples(self, *args, **kwargs):
        """ See abstract method. """
        samples = self.distribution.gen_samples(*args,**kwargs)
        mimic_samples = self._tf_to_mimic_samples(samples)
        return mimic_samples

    def set_dimension(self, dimension):
        """ See abstract method. """
        m = self.mu[0]
        c = self.sigma[0,0]
        expected_cov = c*eye(int(self.d))
        if not (all(self.mu==m) and (self.sigma==expected_cov).all()):
            raise DimensionError('In order to change dimension of Gaussian measure '+\
                'mean (mu) must be all the same and covariance must be a scaler times I')
        self.distribution.set_dimension(dimension)
        self.d = dimension
        self.mu = tile(m,int(self.d))
        self.sigma = c*eye(int(self.d))
        self._assemble()
    
    def plot(self, dim_x=0, dim_y=1, n=2**7, point_size=5, color='c', show=True, out=None):
        """
        Make a scatter plot from samples. Requires dimension >= 2. 

        Args:
            dim_x (int): index of first dimension to be plotted on horizontal axis. 
            dim_y (int): index of the second dimension to be plotted on vertical axis.
            n (int): number of samples to draw as self.gen_samples(n)
            point_size (int): ax.scatter(...,s=point_size)
            color (str): ax.scatter(...,color=color)
            show (bool): show plot or not? 
            out (str): file name to output image. If None, the image is not output

        Return: 
            tuple: fig,ax from `fig,ax = matplotlib.pyplot.subplots(...)`
        """
        if self.dimension < 2:
            raise ParameterError('Plotting a Gaussian instance requires dimension >=2. ')
        x = self.gen_samples(n)
        from matplotlib import pyplot
        pyplot.rc('font', size=16)
        pyplot.rc('legend', fontsize=16)
        pyplot.rc('figure', titlesize=16)
        pyplot.rc('axes', titlesize=16, labelsize=16)
        pyplot.rc('xtick', labelsize=16)
        pyplot.rc('ytick', labelsize=16)
        fig,ax = pyplot.subplots()
        ax.set_xlabel('$x_{i,%d}$'%dim_x)
        ax.set_ylabel('$x_{i,%d}$'%dim_y)
        ax.scatter(x[:,dim_x],x[:,dim_y],color=color,s=point_size)
        s = '$2^{%d}$'%log2(n) if log2(n)%1==0 else '%d'%n
        ax.set_title(s+' Gaussian Samples')
        fig.tight_layout()
        if out: pyplot.savefig(out,dpi=250)
        if show: pyplot.show()
        return fig,ax