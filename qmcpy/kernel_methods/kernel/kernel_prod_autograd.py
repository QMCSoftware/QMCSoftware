
from ._kernel_prod import _KernelProd
from ...discrete_distribution import IIDStdUniform
import numpy as np 
import itertools

class _KernelProdAutoGrad(_KernelProd):
    DEFAULTALPHA = None
    def parsed_call(self, x1, x2, n1, n2, beta1s, beta2s, m1, m2, c1s, c2s):
        try:
            import torch 
        except: 
            raise Exception("_AutoGradKernel requires torch for automatic differentiation")
        xmat1 = [self.npt.tile(x1[:,None,j],(1,n2)) for j in range(self.d)]
        xmat2 = [self.npt.tile(x2[None,:,j],(n1,1)) for j in range(self.d)]
        for j in range(self.d):
            if self.npt==np:
                xmat1[j] = torch.from_numpy(xmat1[j]).float() 
                xmat2[j] = torch.from_numpy(xmat2[j]).float() 
            xmat1[j].requires_grad_()
            xmat2[j].requires_grad_()
        y = self.scale*torch.ones((n1,n2),requires_grad=True)
        for j in range(self.d):
            y = y*self._1d_to_prod(xmat1[j],xmat2[j],self.lengthscales[j])
        grad_outputs = torch.ones((n1,n2),dtype=torch.float)
        v = torch.zeros((n1,n2),requires_grad=False)
        for ell1,ell2 in itertools.product(range(m1),range(m2)):
            yc = y.clone()
            for j in range(self.d):
                for _ in range(beta1s[ell1,j]):
                    yc = torch.autograd.grad(yc,xmat1[j],create_graph=True,grad_outputs=grad_outputs)[0]
                for _ in range(beta2s[ell2,j]):
                    yc = torch.autograd.grad(yc,xmat2[j],create_graph=True,grad_outputs=grad_outputs)[0]
            v = v+c1s[ell1]*c2s[ell2]*yc.detach()
        if self.npt==np: 
            v = v.numpy().astype(np.float64) 
        return v

class KernelGaussian(_KernelProdAutoGrad):
    """ Gaussian kernel 
    
    >>> d = 3
    >>> x = IIDStdUniform(d,seed=7).gen_samples(8)
    >>> k = KernelGaussian(d)
    >>> with np.printoptions(precision=2):
    ...     k(x,x)
    array([[1.  , 0.64, 0.55, 0.85, 0.77, 0.51, 0.43, 0.56],
           [0.64, 1.  , 0.57, 0.54, 0.44, 0.45, 0.58, 0.72],
           [0.55, 0.57, 1.  , 0.63, 0.5 , 0.97, 0.55, 0.83],
           [0.85, 0.54, 0.63, 1.  , 0.5 , 0.6 , 0.27, 0.72],
           [0.77, 0.44, 0.5 , 0.5 , 1.  , 0.5 , 0.63, 0.34],
           [0.51, 0.45, 0.97, 0.6 , 0.5 , 1.  , 0.49, 0.74],
           [0.43, 0.58, 0.55, 0.27, 0.63, 0.49, 1.  , 0.38],
           [0.56, 0.72, 0.83, 0.72, 0.34, 0.74, 0.38, 1.  ]])
    """
    def _1d_to_prod(self, xmat1j, xmat2j, lengthscale):
        import torch 
        return torch.exp(-(xmat1j-xmat2j)**2/lengthscale)