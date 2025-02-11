
from ._kernel_prod import _KernelProd
from ...discrete_distribution import IIDStdUniform
import numpy as np 
import itertools

class _KernelProdAutoGrad(_KernelProd):
    DEFAULTALPHA = np.nan
    def parsed_call(self, x1, x2, n1, n2, beta1s, beta2s, m1, m2, c1s, c2s):
        try:
            import torch 
        except: 
            raise Exception("_AutoGradKernel requires torch for automatic differentiation")
        xmat1 = [self.npt.tile(x1[:,None,j],(1,n2)) for j in range(self.d)]
        xmat2 = [self.npt.tile(x2[None,:,j],(n1,1)) for j in range(self.d)]
        for j in range(self.d):
            if self.npt==np:
                xmat1[j] = torch.from_numpy(xmat1[j])
                xmat2[j] = torch.from_numpy(xmat2[j])
            xmat1[j].requires_grad_()
            xmat2[j].requires_grad_()
        y = self.scale*torch.ones((n1,n2),requires_grad=True,**self.ckwargs)
        for j in range(self.d):
            y = y*self._1d_to_prod(xmat1[j],xmat2[j],self.lengthscales[j])
        grad_outputs = torch.ones((n1,n2),dtype=torch.float,**self.ckwargs)
        v = torch.zeros((n1,n2),requires_grad=False,**self.ckwargs)
        for ell1,ell2 in itertools.product(range(m1),range(m2)):
            yc = y.clone()
            for j in range(self.d):
                for _ in range(beta1s[ell1,j]):
                    yc = torch.autograd.grad(yc,xmat1[j],create_graph=True,grad_outputs=grad_outputs)[0]
                for _ in range(beta2s[ell2,j]):
                    yc = torch.autograd.grad(yc,xmat2[j],create_graph=True,grad_outputs=grad_outputs)[0]
            v = v+c1s[ell1]*c2s[ell2]*yc#.detach()
        if self.npt==np: 
            v = v.detach().cpu().numpy().astype(np.float64) 
        return v

class KernelGaussian(_KernelProdAutoGrad):
    """ Gaussian kernel 
    
    >>> import torch
    >>> d = 3
    >>> x = torch.from_numpy(IIDStdUniform(d,seed=7).gen_samples(8))
    >>> k = KernelGaussian(d,torchify=True)
    >>> with np.printoptions(precision=2):
    ...     k(x,x)
    tensor([[1.0000, 0.7980, 0.7412, 0.9196, 0.8771, 0.7125, 0.6546, 0.7469],
            [0.7980, 1.0000, 0.7575, 0.7362, 0.6654, 0.6713, 0.7642, 0.8466],
            [0.7412, 0.7575, 1.0000, 0.7915, 0.7102, 0.9872, 0.7444, 0.9120],
            [0.9196, 0.7362, 0.7915, 1.0000, 0.7092, 0.7737, 0.5166, 0.8509],
            [0.8771, 0.6654, 0.7102, 0.7092, 1.0000, 0.7067, 0.7925, 0.5816],
            [0.7125, 0.6713, 0.9872, 0.7737, 0.7067, 1.0000, 0.6982, 0.8583],
            [0.6546, 0.7642, 0.7444, 0.5166, 0.7925, 0.6982, 1.0000, 0.6141],
            [0.7469, 0.8466, 0.9120, 0.8509, 0.5816, 0.8583, 0.6141, 1.0000]],
           dtype=torch.float64)
    """
    def _1d_to_prod(self, xmat1j, xmat2j, lengthscale):
        import torch 
        return torch.exp(-(xmat1j-xmat2j)**2/(2*lengthscale))