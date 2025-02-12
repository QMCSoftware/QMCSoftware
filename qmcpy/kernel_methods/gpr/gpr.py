from ...discrete_distribution import Lattice,DigitalNetB2
from ...discrete_distribution._discrete_distribution import DiscreteDistribution
from ..kernel import KernelGaussian,KernelShiftInvar,KernelDigShiftInvar
from ..gram_matrix import GramMatrix,FastGramMatrixLattice,FastGramMatrixDigitalNetB2
import numpy as np 

class _GPR(object):
    def _check_torch(self):
        try:
            import torch 
        except ImportError:
            raise Exception("Gaussian Process Regression requires torch for autograd of hyperparameters")
    def __init__(self, lbetas, lcs, alpha, kernel_class, kernel_kwargs={}):
        import torch 
        assert hasattr(self,"x") and isinstance(self.x,torch.Tensor) and self.x.ndim==2
        if hasattr(self,"_x"): assert isinstance(self._x,torch.Tensor) and self._x.ndim==2 and self.x.device==self._x.device
        self.n = self.x.size(0)
        self.d = self.x.size(1) 
        self.device = self.x.device
        self.alpha = alpha 
        self.kernel_class = kernel_class
        self.kernel_kwargs = kernel_kwargs
        self.lbetas = lbetas 
        self.lcs = lcs
    def fit(self, 
            yf, 
            x_test, 
            y_test,
            opt_steps = 25, 
            lengthscales = None, 
            global_scale = None, 
            noises = None, 
            opt_lengthscales = True, 
            opt_global_scale = True, 
            opt_noises = True,
            optimizer_init = None,
            verbose = True,
            verbose_indent = 4,
        ):
        import torch 
        assert yf.ndim==2 and yf.size(0)==self.n and yf.device==self.x.device
        assert y_test.ndim==1 and x_test.ndim==2 and y_test.size(0)==x_test.size(0) and x_test.size(1)==self.d and y_test.device==x_test.device==self.device
        d_out = yf.size(1) 
        yflat = yf.T.flatten()
        if lengthscales is None:
            lengthscales =  torch.ones(self.d,device=self.device)
        assert isinstance(lengthscales,torch.Tensor) and lengthscales.device==self.device and lengthscales.shape==(self.d,) and (lengthscales>0).all()
        if global_scale is None: 
            global_scale = torch.ones(1,device=self.device)
        assert isinstance(global_scale,torch.Tensor) and global_scale.device==self.device and global_scale.shape==(1,) and (global_scale>0).all()
        if noises is None: 
            noises = 1e-8*torch.ones(d_out,device=self.device) 
        assert isinstance(noises,torch.Tensor) and noises.device==self.device and noises.shape==(d_out,) and (noises>0).all()
        log10_lengthscales = torch.log10(lengthscales)
        log10_global_scale = torch.log10(global_scale)
        log10_noises = torch.log10(noises)
        assert opt_lengthscales or opt_global_scale or opt_noises, "need to optimize lengthscales and/or global_scale and/or noises"
        params_to_opt = [] 
        if opt_lengthscales:
            log10_lengthscales = torch.nn.Parameter(log10_lengthscales)
            params_to_opt.append(log10_lengthscales)
        if opt_global_scale:
            log10_global_scale = torch.nn.Parameter(log10_global_scale)
            params_to_opt.append(log10_global_scale)
        if opt_noises:
            log10_noises = torch.nn.Parameter(log10_noises)
            params_to_opt.append(log10_noises)
        if optimizer_init is None: optimizer_init = lambda params: torch.optim.Adam(params,lr=1e-1,amsgrad=True)
        assert callable(optimizer_init)
        optimizer = optimizer_init(params_to_opt)
        lengthscales_hist = torch.empty((opt_steps+1,self.d))
        global_scale_hist = torch.empty(opt_steps+1)
        noises_hist = torch.empty((opt_steps+1,d_out))
        mll_hist = torch.empty(opt_steps+1)
        l2rerr_hist = torch.empty(opt_steps+1)
        if verbose:
            _s = "%15s | %-15s %-15s | lengthscales, global_scale, noises"%("iter of %d"%opt_steps,"MLL","L2RError")
            print(" "*verbose_indent+_s)
            print(" "*verbose_indent+"~"*len(_s))
        for i in range(opt_steps+1):
            lengthscales = 10**log10_lengthscales
            global_scale = 10**log10_global_scale
            noises = 10**log10_noises
            kernel = self.kernel_class(self.d,alpha=self.alpha,lengthscales=lengthscales,scale=global_scale,torchify=True,device=self.device,**self.kernel_kwargs)
            self.gm = self._set_gram_matrix(kernel, noises)
            self.coeffs = self.gm.solve(yflat)
            mll = yflat[None,:]@self.coeffs+self.gm.logdet()
            k_left = self.gm.get_new_left_full_gram_matrix(x_test)
            yhat = k_left@self.coeffs
            lengthscales_hist[i] = lengthscales.detach().cpu()
            global_scale_hist[i] = global_scale.detach().cpu()
            noises_hist[i] = noises.detach().cpu()
            mll_hist[i] = mll.detach()
            l2rerr_hist[i] = (torch.linalg.norm(y_test-yhat)/torch.linalg.norm(y_test)).detach()
            if verbose and (i%verbose==0 or i==opt_steps):
                _s = "%15d | %-15.2e %-15.2e | "%(i,mll_hist[i],l2rerr_hist[i])
                with np.printoptions(formatter={"float":lambda x: "%.2e"%x}):
                    _s += "%s\t%s\t%s"%(str(lengthscales_hist[i].numpy()),str(global_scale_hist[i,None].numpy()),str(noises_hist[i].numpy()))
                print(" "*verbose_indent+_s)
            if i==opt_steps: break
            mll.backward()
            optimizer.step()
        data = {
            "lengthscales": lengthscales_hist,
            "global_scale": global_scale_hist,
            "noises": noises_hist,
            "mll": mll_hist,
            "l2rerr": l2rerr_hist}
        return data 
    
class GPR(_GPR):
    def __init__(self, dd_obj_or_x, n=None, lbetas=0, lcs=1, device="cpu", kernel_class=KernelGaussian, alpha=None):
        self._check_torch()
        import torch
        if isinstance(dd_obj_or_x,DiscreteDistribution):
            assert isinstance(n,int), "n is required if dd_obj_or_x is a DiscreteDistribution"
            self.x = torch.from_numpy(dd_obj_or_x.gen_samples(n)).to(device)
        else:
            self.x = dd_obj_or_x
        super(GPR,self).__init__(lbetas,lcs,alpha,kernel_class)
    def _set_gram_matrix(self, kernel, noises):
        return GramMatrix(kernel,self.x,self.x,lbeta1s=self.lbetas,lbeta2s=self.lbetas,adaptive_noise=False,noise=noises)

class FGPRLattice(_GPR):
    def __init__(self, dd_obj, n, lbetas=0, lcs=1, device="cpu", alpha=None):
        self._check_torch()
        import torch
        assert isinstance(dd_obj,Lattice)
        self.dd_obj = dd_obj
        self.x = self._x = torch.from_numpy(self.dd_obj.gen_samples(n)).to(device)
        super(FGPRLattice,self).__init__(lbetas,lcs,alpha,kernel_class=KernelShiftInvar)
    def _set_gram_matrix(self, kernel, noises):
        return FastGramMatrixLattice(kernel,self.dd_obj,self.n,self.n,lbeta1s=self.lbetas,lbeta2s=self.lbetas,adaptive_noise=False,noise=noises,_pregenerated_x__x=(self._x,self.x))

class FGPRDigitalNetB2(_GPR):
    def __init__(self, dd_obj, n, lbetas=0, lcs=1, device="cpu", alpha=None):
        self._check_torch()
        import torch
        assert isinstance(dd_obj,DigitalNetB2)
        self.dd_obj = dd_obj
        self._x = torch.from_numpy(self.dd_obj.gen_samples(n,return_binary=True).astype(np.int64)).to(device)
        self.x = self._x*2**(-dd_obj.t_lms)
        super(FGPRDigitalNetB2,self).__init__(lbetas,lcs,alpha,kernel_class=KernelDigShiftInvar,kernel_kwargs={"t":dd_obj.t_lms})
    def _set_gram_matrix(self, kernel, noises):
        return FastGramMatrixDigitalNetB2(kernel,self.dd_obj,self.n,self.n,lbeta1s=self.lbetas,lbeta2s=self.lbetas,adaptive_noise=False,noise=noises,_pregenerated_x__x=(self._x,self.x))
