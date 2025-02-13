from ...discrete_distribution import Lattice,DigitalNetB2,IIDStdUniform,Halton
from ...discrete_distribution._discrete_distribution import DiscreteDistribution
from ..kernel import KernelGaussian,KernelShiftInvar,KernelDigShiftInvar
from ..gram_matrix import GramMatrix,FastGramMatrixLattice,FastGramMatrixDigitalNetB2
import numpy as np 
import timeit

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
            x_test = None, 
            y_test = None,
            opt_steps = 25, 
            lengthscales = None, 
            global_scale = None, 
            noises = None, 
            opt_lengthscales = True, 
            opt_global_scale = True, 
            opt_noises = False,
            lb_lengthscale = 0, 
            lb_global_scale = 0, 
            lb_noises = 1e-16,
            optimizer_init = None,
            use_scheduler = False,
            verbose = True,
            verbose_indent = 4,
        ):
        import torch 
        t0 = timeit.default_timer()
        assert yf.ndim==2 and yf.size(0)==self.n and yf.device==self.x.device
        test = x_test is not None or y_test is not None
        if test:
            assert y_test.ndim==1 and x_test.ndim==2 and y_test.size(0)==x_test.size(0) and x_test.size(1)==self.d and y_test.device==x_test.device==self.device
        d_out = yf.size(1) 
        yflat = yf.T.flatten()
        if lengthscales is None:
            lengthscales =  lb_lengthscale+torch.ones(self.d,device=self.device)
        assert isinstance(lengthscales,torch.Tensor) and lengthscales.device==self.device and lengthscales.shape==(self.d,) and (lengthscales>lb_lengthscale).all()
        if global_scale is None: 
            global_scale = lb_global_scale+torch.ones(1,device=self.device)
        assert isinstance(global_scale,torch.Tensor) and global_scale.device==self.device and global_scale.shape==(1,) and (global_scale>lb_global_scale).all()
        if noises is None: 
            noises = lb_noises+1e-8*torch.ones(d_out,device=self.device) 
        assert isinstance(noises,torch.Tensor) and noises.device==self.device and noises.shape==(d_out,) and (noises>lb_noises).all()
        log10_lengthscales = torch.log10(lengthscales-lb_lengthscale)
        log10_global_scale = torch.log10(global_scale-lb_global_scale)
        log10_noises = torch.log10(noises-lb_noises)
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
        if optimizer_init is None: optimizer_init = lambda params: torch.optim.Rprop(params,lr=.1)
        assert callable(optimizer_init)
        optimizer = optimizer_init(params_to_opt)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=1/2,patience=0,threshold=1e-2,threshold_mode="rel")
        lengthscales_hist = torch.empty((opt_steps+1,self.d))
        global_scale_hist = torch.empty(opt_steps+1)
        noises_hist = torch.empty((opt_steps+1,d_out))
        mll_hist = torch.empty(opt_steps+1)
        if test:
            l2rerr_hist = torch.empty(opt_steps+1)
        times_hist = torch.empty(opt_steps+1)
        if verbose:
            _s = "%15s | %-15s %-15s | %-15s | lengthscales, global_scale, noises"%("iter of %d"%opt_steps,"MLL","L2RError","lr")
            print(" "*verbose_indent+_s)
            print(" "*verbose_indent+"~"*len(_s))
        for i in range(opt_steps+1):
            if i==opt_steps:
                for param in params_to_opt:
                    param.requires_grad_(False)
            lengthscales = 10**log10_lengthscales+lb_lengthscale
            global_scale = 10**log10_global_scale+lb_global_scale
            noises = 10**log10_noises+lb_noises
            optimizer.zero_grad()
            kernel = self.kernel_class(self.d,alpha=self.alpha,lengthscales=lengthscales,scale=global_scale,torchify=True,device=self.device,requires_grad=i<opt_steps,**self.kernel_kwargs)
            self.gm = self._set_gram_matrix(kernel, noises)
            self.coeffs = self.gm.solve(yflat)
            mll = yflat[None,:]@self.coeffs+self.gm.logdet()
            lengthscales_hist[i] = lengthscales.detach().cpu()
            global_scale_hist[i] = global_scale.detach().cpu()
            noises_hist[i] = noises.detach().cpu()
            mll_hist[i] = mll.detach()
            if test:
                yhat = self.post_mean(x_test)
                l2rerr_hist[i] = (torch.linalg.norm(y_test-yhat)/torch.linalg.norm(y_test)).detach()
            if verbose and (i%verbose==0 or i==opt_steps):
                _s = "%15d | %-15.2e %-15.2e | %-15.2e | "%(i,mll_hist[i],l2rerr_hist[i] if test else np.nan,scheduler.get_last_lr()[0])
                with np.printoptions(formatter={"float":lambda x: "%.2e"%x}):
                    _s += "%s\t%s\t%s"%(str(lengthscales_hist[i].numpy()),str(global_scale_hist[i,None].numpy()),str(noises_hist[i].numpy()))
                print(" "*verbose_indent+_s)
            times_hist[i] = timeit.default_timer()-t0
            if i==opt_steps: break
            mll.backward()
            #if i==0 or mll<(2*mll_hist[i-1]):
            optimizer.step()
            if use_scheduler:
                scheduler.step(mll)
        data = {
            "lengthscales": lengthscales_hist,
            "global_scale": global_scale_hist,
            "noises": noises_hist,
            "mll": mll_hist,
            "times": times_hist,
        }
        if test:
            data["l2rerr"] = l2rerr_hist
        return data 
    def post_mean(self, x, betas=0, cs=1):
        assert x.ndim==2 and x.device==self.x.device and x.size(1)==self.d
        k_left = self.gm.get_new_left_full_gram_matrix(x,betas,cs)
        yhat = k_left@self.coeffs
        return yhat 
    def post_var(self, x, betas=0, cs=1):
        import torch
        xb = self.gm._convert_x_to__x(x)
        kvec = self.gm.kernel_obj(xb,xb,betas,betas,cs,cs,diag_only=True)
        k_left = self.gm.get_new_left_full_gram_matrix(x,betas,cs)
        k_right = k_left.T 
        pvar = kvec-torch.einsum("ik,ki->i",k_left,self.gm.solve(k_right))
        return pvar
    def post_cov(self, x, betas=0, cs=1):
        xb = self.gm._convert_x_to__x(x)
        k = self.gm.kernel_obj(xb,xb,betas,betas,cs,cs)
        k_left = self.gm.get_new_left_full_gram_matrix(x,betas,cs)
        k_right = k_left.T
        pcov = k-k_left@self.gm.solve(k_right)
        return pcov
    def post_cov_sep(self, x1, x2, beta1s=0, beta2s=0, c1s=1, c2s=1):
        xb1 = self.gm._convert_x_to__x(x1)
        xb2 = self.gm._convert_x_to__x(x2)
        k = self.gm.kernel_obj(xb1,xb2,beta1s,beta2s,c1s,c2s)
        k_left = self.gm.get_new_left_full_gram_matrix(x1,beta1s,c1s)
        k_right = self.gm.get_new_left_full_gram_matrix(x2,beta2s,c2s).T
        pcov = k-k_left@self.gm.solve(k_right)
        return pcov  

class GPR(_GPR):
    """
    >>> import torch
    >>> d = 2
    >>> lbetas = [torch.zeros((1,d),dtype=int)]+[ej for ej in torch.eye(d,dtype=int)]
    >>> iid = IIDStdUniform(d,seed=7)
    >>> gp = GPR(iid,n=32,lbetas=lbetas)
    >>> yf = torch.vstack([
    ...     torch.cos(gp.x[:,0])*torch.sin(gp.x[:,1]),
    ...     -torch.sin(gp.x[:,0])*torch.sin(gp.x[:,1]),
    ...     torch.cos(gp.x[:,0])*torch.cos(gp.x[:,1]),
    ... ]).T
    >>> x_test = torch.from_numpy(Halton(d,seed=17).gen_samples(64))
    >>> y_test = torch.cos(x_test[:,0])*torch.sin(x_test[:,1])
    >>> data = gp.fit(yf,x_test,y_test,opt_steps=10,verbose=False)
    >>> pmean = gp.post_mean(gp.x)
    >>> pmean.shape 
    torch.Size([32])
    >>> torch.allclose(pmean,yf[:,0],atol=1e-5,rtol=0)
    True
    >>> pvar = gp.post_var(x_test)
    >>> pvar.shape 
    torch.Size([64])
    >>> pcov = gp.post_cov(x_test)
    >>> pcov.shape
    torch.Size([64, 64])
    >>> torch.allclose(pcov.diagonal(),pvar)
    True
    >>> pcov_sep = gp.post_cov_sep(x_test,gp.x)
    >>> pcov_sep.shape
    torch.Size([64, 32])
    >>> pvar = gp.post_var(gp.x)
    >>> torch.allclose(pvar,torch.zeros_like(pvar),atol=1e-6,rtol=0)
    True
    """
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
    """
    >>> import torch
    >>> d = 2
    >>> lbetas = [torch.zeros((1,d),dtype=int)]+[ej for ej in torch.eye(d,dtype=int)]
    >>> lattice = Lattice(d,seed=7)
    >>> gp = FGPRLattice(lattice,n=32,lbetas=lbetas)
    >>> yf = torch.vstack([
    ...     torch.cos(gp.x[:,0])*torch.sin(gp.x[:,1]),
    ...     -torch.sin(gp.x[:,0])*torch.sin(gp.x[:,1]),
    ...     torch.cos(gp.x[:,0])*torch.cos(gp.x[:,1]),
    ... ]).T
    >>> x_test = torch.from_numpy(Halton(d,seed=17).gen_samples(64))
    >>> y_test = torch.cos(x_test[:,0])*torch.sin(x_test[:,1])
    >>> data = gp.fit(yf,x_test,y_test,opt_steps=10,verbose=False)
    >>> pmean = gp.post_mean(gp.x)
    >>> pmean.shape 
    torch.Size([32])
    >>> torch.allclose(pmean,yf[:,0],atol=1e-5,rtol=0)
    True
    >>> pvar = gp.post_var(x_test)
    >>> pvar.shape 
    torch.Size([64])
    >>> pcov = gp.post_cov(x_test)
    >>> pcov.shape
    torch.Size([64, 64])
    >>> torch.allclose(pcov.diagonal(),pvar)
    True
    >>> pcov_sep = gp.post_cov_sep(x_test,gp.x)
    >>> pcov_sep.shape
    torch.Size([64, 32])
    >>> pvar = gp.post_var(gp.x)
    >>> torch.allclose(pvar,torch.zeros_like(pvar),atol=1e-6,rtol=0)
    True
    """
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
    """
    >>> import torch
    >>> d = 2
    >>> lbetas = [torch.zeros((1,d),dtype=int)]+[ej for ej in torch.eye(d,dtype=int)]
    >>> dnb2 = DigitalNetB2(d,seed=7)
    >>> gp = FGPRDigitalNetB2(dnb2,n=32,lbetas=lbetas,alpha=4)
    >>> yf = torch.vstack([
    ...     torch.cos(gp.x[:,0])*torch.sin(gp.x[:,1]),
    ...     -torch.sin(gp.x[:,0])*torch.sin(gp.x[:,1]),
    ...     torch.cos(gp.x[:,0])*torch.cos(gp.x[:,1]),
    ... ]).T
    >>> x_test = torch.from_numpy(Halton(d,seed=17).gen_samples(64))
    >>> y_test = torch.cos(x_test[:,0])*torch.sin(x_test[:,1])
    >>> data = gp.fit(yf,x_test,y_test,opt_steps=10,verbose=False)
    >>> pmean = gp.post_mean(gp.x)
    >>> pmean.shape 
    torch.Size([32])
    >>> torch.allclose(pmean,yf[:,0],atol=1e-5,rtol=0)
    True
    >>> pvar = gp.post_var(x_test)
    >>> pvar.shape 
    torch.Size([64])
    >>> pcov = gp.post_cov(x_test)
    >>> pcov.shape
    torch.Size([64, 64])
    >>> torch.allclose(pcov.diagonal(),pvar)
    True
    >>> pcov_sep = gp.post_cov_sep(x_test,gp.x)
    >>> pcov_sep.shape
    torch.Size([64, 32])
    >>> pvar = gp.post_var(gp.x)
    >>> torch.allclose(pvar,torch.zeros_like(pvar),atol=1e-6,rtol=0)
    True
    """
    def __init__(self, dd_obj, n, lbetas=0, lcs=1, device="cpu", alpha=None):
        self._check_torch()
        import torch
        assert isinstance(dd_obj,DigitalNetB2)
        self.dd_obj = dd_obj
        self._x = torch.from_numpy(self.dd_obj.gen_samples(n,return_binary=True).astype(np.int64)).to(device)
        self.x = (self._x*2**(-dd_obj.t_lms)).to(float)
        super(FGPRDigitalNetB2,self).__init__(lbetas,lcs,alpha,kernel_class=KernelDigShiftInvar,kernel_kwargs={"t":dd_obj.t_lms})
    def _set_gram_matrix(self, kernel, noises):
        return FastGramMatrixDigitalNetB2(kernel,self.dd_obj,self.n,self.n,lbeta1s=self.lbetas,lbeta2s=self.lbetas,adaptive_noise=False,noise=noises,_pregenerated_x__x=(self._x,self.x))
