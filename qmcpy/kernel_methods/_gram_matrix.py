import scipy.linalg

class _GramMatrix(object):
    def __init__(self, kernel_obj, noise, lbeta1s, lbeta2s, lc1s, lc2s):
        self.kernel_obj = kernel_obj
        self.d = self.kernel_obj.d
        self.npt = self.kernel_obj.npt
        self.torchify = self.kernel_obj.torchify
        self.noise = noise
        assert isinstance(self.noise,float) and self.noise>0.
        if isinstance(lbeta1s,int): lbeta1s = [lbeta1s*self.npt.ones((1,self.d),dtype=int)]
        elif not isinstance(lbeta1s,list): lbeta1s = [self.npt.atleast_2d(lbeta1s)]
        if isinstance(lbeta2s,int): lbeta2s = [lbeta2s*self.npt.ones((1,self.d),dtype=int)]
        elif not isinstance(lbeta2s,list): lbeta2s = [self.npt.atleast_2d(lbeta2s)]
        self.lbeta1s = [self.npt.atleast_2d(beta1s) for beta1s in lbeta1s]
        self.lbeta2s = [self.npt.atleast_2d(beta2s) for beta2s in lbeta2s]
        self.t1 = len(self.lbeta1s)
        self.t2 = len(self.lbeta2s)
        self.m1 = self.npt.array([len(beta1s) for beta1s in self.lbeta1s],dtype=int)
        self.m2 = self.npt.array([len(beta2s) for beta2s in self.lbeta2s],dtype=int)
        assert isinstance(self.lbeta1s,list) and all(self.lbeta1s[tt1].shape==(self.m1[tt1],self.d) for tt1 in range(self.t1))
        assert isinstance(self.lbeta2s,list) and all(self.lbeta2s[tt2].shape==(self.m2[tt2],self.d) for tt2 in range(self.t2))
        if isinstance(lc1s,float): lc1s = [lc1s*self.npt.ones(self.m1[tt1]) for tt1 in range(self.t1)]
        elif not isinstance(lc1s,list): lc1s = [lc1s]
        if isinstance(lc2s,float): lc2s = [lc2s*self.npt.ones(self.m2[tt2]) for tt2 in range(self.t2)]
        elif not isinstance(lc2s,list): lc2s = [lc2s]
        self.lc1s = [self.npt.atleast_1d(c1s) for c1s in lc1s] 
        self.lc2s = [self.npt.atleast_1d(c2s) for c2s in lc2s]
        self.lc1s_og = [self.npt.atleast_1d(c1s) for c1s in lc1s] 
        self.lc2s_og = [self.npt.atleast_1d(c2s) for c2s in lc2s]
        assert isinstance(self.lc1s,list) and all(self.lc1s[tt1].shape==(self.m1[tt1],) for tt1 in range(self.t1))
        assert isinstance(self.lc2s,list) and all(self.lc2s[tt2].shape==(self.m2[tt2],) for tt2 in range(self.t2))
        if self.torchify:
            import torch 
            self.cho_solve = lambda l,b: torch.cholesky_solve(b,l,upper=False)
        else:
            self.cho_solve = lambda l,b: scipy.linalg.cho_solve((l,True),b)
    def cholesky(self, mat):
        try:
            l_chol = self.npt.linalg.cholesky(mat)
        except:
            raise Exception("Cholesky not positive definite, try increasing the noise")
        return l_chol
    def _set_invertible_conds(self, invertible_conds):
        self.invertible = all(cond for cond,error_msg in invertible_conds)
        self.invertible_error_msg = "\n\n"+"\n\n".join(error_msg for cond,error_msg in invertible_conds if not cond)
