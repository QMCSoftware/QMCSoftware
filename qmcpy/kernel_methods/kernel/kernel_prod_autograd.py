
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
    
    >>> import torch
    >>> d = 3
    >>> x = torch.from_numpy(IIDStdUniform(d,seed=7).gen_samples(8))
    >>> k = KernelGaussian(d,torchify=True)
    >>> with np.printoptions(precision=2):
    ...     k(x,x)
    tensor([[1.0000, 0.6369, 0.5493, 0.8457, 0.7693, 0.5076, 0.4285, 0.5578],
            [0.6369, 1.0000, 0.5738, 0.5420, 0.4427, 0.4507, 0.5841, 0.7167],
            [0.5493, 0.5738, 1.0000, 0.6265, 0.5043, 0.9745, 0.5541, 0.8318],
            [0.8457, 0.5420, 0.6265, 1.0000, 0.5029, 0.5986, 0.2669, 0.7240],
            [0.7693, 0.4427, 0.5043, 0.5029, 1.0000, 0.4994, 0.6281, 0.3383],
            [0.5076, 0.4507, 0.9745, 0.5986, 0.4994, 1.0000, 0.4875, 0.7366],
            [0.4285, 0.5841, 0.5541, 0.2669, 0.6281, 0.4875, 1.0000, 0.3771],
            [0.5578, 0.7167, 0.8318, 0.7240, 0.3383, 0.7366, 0.3771, 1.0000]],
           dtype=torch.float64)
    """
    def _1d_to_prod(self, xmat1j, xmat2j, lengthscale):
        import torch 
        return torch.exp(-(xmat1j-xmat2j)**2/(2*lengthscale))