from qmcpy.true_measure._true_measure import TrueMeasure
from qmcpy.util import DimensionError
from ..discrete_distribution import Sobol
from numpy import *


class Kumaraswamy(TrueMeasure):
    """
    For $\\boldsymbol{a},\\boldsymbol{b}>0$ we have

    PDF: $f(\\boldsymbol{x}) = \\prod_{j=1}^d [a_j b_j x_j^{a_j-1}(1-x_j^{a_j})^{b_j-1}]$

    CDF: $F(\\boldsymbol{x}) = \\prod_{j=1}^d [1-(1-x_j^{a_j})^{b_j}]$

    Inverse CDF: $\\Psi_j(x_j) = (1-(1-x_j)^{1/b_j})^{1/a_j}$
    """

    def __init__(self, sampler, a=2, b=2):
        """
        Args:
            sampler (DiscreteDistribution/TrueMeasure): A 
                discrete distribution from which to transform samples or a
                true measure by which to compose a transform 
            a (ndarray): a 
            b (ndarray): b
        """
        self.parameters = ['a', 'b']
        self.domain = array([[0,1]])
        self.range = array([[0,1]])
        self._parse_sampler(sampler)
        self.a = a
        self.b = b
        if isscalar(self.a):
            a = tile(self.a,self.d)
        if isscalar(self.b):
            b = tile(self.b,self.d)
        self.alpha = array(a)
        self.beta = array(b)
        if len(self.alpha)!=self.d or len(self.beta)!=self.d:
            raise DimensionError('a and b must be scalar or have length equal to dimension.')
        super(Kumaraswamy,self).__init__() 

    def _transform(self, x):
        return (1-(1-x)**(1/self.beta))**(1/self.alpha)

    def _jacobian(self, x):
        return prod( (1-(1-x)**(1/self.beta))**(1/self.alpha-1)*(1-x)**(1/self.beta-1)/(self.alpha*self.beta), 1)
    
    def _weight(self, x):
        return prod( self.alpha*self.beta*x**(self.alpha-1)*(1-x**self.alpha)**(self.beta-1), 1)
    
    def _set_dimension(self, dimension):
        a = self.alpha[0]
        b = self.beta[0]
        if not (all(self.alpha==a) and all(self.beta==b)):
            raise DimensionError('''
                In order to change dimension of a Kumaraswamy measure
                a must all be the same and 
                b must all be the same''')
        self.d = dimension
        self.alpha = tile(a,self.d)
        self.beta = tile(b,self.d)