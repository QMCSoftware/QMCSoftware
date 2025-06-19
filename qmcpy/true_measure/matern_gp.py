from .gaussian import Gaussian
from ..discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution import DigitalNetB2
from ..util import DimensionError, ParameterError
import numpy as np
from scipy.special import kv, gamma
from typing import Union


class MaternGP(Gaussian):
    r"""
    A Gaussian process with Matérn covariance kernel.

    Examples:
        >>> true_measure = MaternGP(DigitalNetB2(dimension=3,seed=7),points=np.linspace(0,1,3)[:,None],nu=3/2,length_scale=[3,4,5],variance=0.01,mean=np.array([.3,.4,.5]))
        >>> true_measure(4)
        array([[0.36584129, 0.41955332, 0.4863312 ],
               [0.27844263, 0.38959308, 0.48847698],
               [0.38043141, 0.50676846, 0.61755968],
               [0.22511911, 0.30830933, 0.40481964]])
        >>> true_measure
        MaternGP (AbstractTrueMeasure)
            mean            [0.3 0.4 0.5]
            covariance      [[0.01  0.01  0.01 ]
                             [0.01  0.01  0.01 ]
                             [0.009 0.01  0.01 ]]
            decomp_type     PCA
        
        With independent replications 

        >>> x = MaternGP(DigitalNetB2(dimension=3,seed=7,replications=2),points=np.linspace(0,1,3)[:,None],nu=3/2,length_scale=[3,4,5],variance=0.01,mean=np.array([.3,.4,.5]))(4)
        >>> x.shape 
        (2, 4, 3)
        >>> x
        array([[[0.32152508, 0.38823324, 0.47330157],
                [0.34399008, 0.45835862, 0.56317366],
                [0.17920615, 0.31871417, 0.44005234],
                [0.52228111, 0.6178775 , 0.70873087]],
        <BLANKLINE>
               [[0.21490091, 0.33078241, 0.45151042],
                [0.40274891, 0.50833272, 0.5996952 ],
                [0.32224364, 0.39269575, 0.46851234],
                [0.31973076, 0.42545283, 0.54711899]]])
    
    **References:**

    1.  [`sklearn.gaussian_process.kernels.Matern`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.MaternGP.html).
    
    2.  [https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function).
    """

    def __init__(self, sampler, points, length_scale=1.0, nu=1.5, variance=1.0, mean=0., nugget=1e-6, decomp_type='PCA'):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution,AbstractTrueMeasure]): Either  
                
                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            points (np.ndarray): The positions of points on a metric space. The array should have shape $(d,k)$ where $d$ is the dimension of the sampler and $k$ is the latent dimension. 
            nu (float): The "smoothness" of the MaternGP function, e.g.,
                
                - $\nu = 1/2$ is equivalent to the absolute exponential kernel, 
                - $\nu = 3/2$ implies a once-differentiable function,
                - $\nu = 5/2$ implies twice differentiability. 
                - as $\nu \to \infty$ the kernel becomes equivalent to the RBF kernel, see [`sklearn.gaussian_process.kernels.RBF`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF). 
                
                Note that when $\nu \notin \{1/2, 3/2, 5/2, \infty \}$ the kernel is around $10$ times slower to evaluate. 
            length_scale (Union[float,np.ndarray]): Determines "peakiness", or how correlated two points are based on their distance.
            variance (float): Global scaling factor.
            mean (Union[float,np.ndarray]): Mean vectorfor multivariante `Gaussian`. 
            nugget (float): Positive nugget to add to diagonal.
            decomp_type (str): Method for decomposition for covariance matrix. Options include
             
                - `'PCA'` for principal component analysis, or 
                - `'Cholesky'` for cholesky decomposition.
        """
        if not (isinstance(sampler, AbstractDiscreteDistribution) or isinstance(sampler, AbstractTrueMeasure)):
            raise ParameterError("sampler input should either be an AbstractDiscreteDistribution or AbstractTrueMeasure.")
        if not (isinstance(points, np.ndarray) and (points.ndim==1 or points.ndim==2)):
            raise ParameterError("points must be a one or two dimensional np.ndarray.")
        if points.ndim==1: points = points[:,None] 
        assert points.ndim==2 and points.shape[0]==sampler.d, "points should be a two dimenssion array with the number of points equal to the dimension of the sampler"
        mean = np.array(mean)
        if mean.size==1: mean = mean.item()*np.ones(sampler.d)
        assert mean.shape==(sampler.d,), "mean should be a length d vector"
        assert np.isscalar(nu) and nu>0, "nu should be a positive scalar"
        length_scale = np.array(length_scale) 
        if length_scale.size==1: length_scale = length_scale.item()*np.ones(sampler.d)
        assert length_scale.shape==(sampler.d,) and (length_scale>0).all(), "length_scale should be a vector with length equal to the dimension of the sampler"
        assert np.isscalar(variance) and variance>0, "length_scale should be a positive scalar"
        assert np.isscalar(nugget) and nugget>0, "nugget should be a positive scalar"
        self.points = points
        self.length_scale = length_scale
        self.nu = nu
        self.variance = variance
        dists = np.linalg.norm(points[...,:,None,:]-points[...,None,:,:],axis=-1)
        if nu==1/2:
            covariance = np.exp(-dists/self.length_scale)
        elif nu==3/2:
            covariance = (1+np.sqrt(3)*dists/self.length_scale)*np.exp(-np.sqrt(3)*dists/self.length_scale)
        elif nu==5/2:
            covariance = ((1+np.sqrt(5)*dists/self.length_scale+5*dists**2/(3*self.length_scale**2))*np.exp(-np.sqrt(5)*dists/self.length_scale))
        elif nu==np.inf:
            covariance = np.exp(-dists**2/(2*self.length_scale**2))
        else:
            k = np.sqrt(2*nu)*dists/self.length_scale
            covariance = 2**(1-nu)/gamma(nu)*k**nu*kv(nu,k)
        covariance = variance*covariance+nugget*np.eye(sampler.d)
        super().__init__(sampler, mean=mean, covariance=covariance, decomp_type=decomp_type)

    def _spawn(self, sampler):
        return MaternGP(sampler, self.points, length_scale=self.length_scale, nu=self.nu, variance=self.variance, 
                      mean=self.mean, decomp_type=self.decomp_type)
        
