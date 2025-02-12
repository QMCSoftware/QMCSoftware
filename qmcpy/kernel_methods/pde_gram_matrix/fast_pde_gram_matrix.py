from ._pde_gram_matrix import _PDEGramMatrix
from ...discrete_distribution import Lattice,DigitalNetB2
from ..kernel import KernelShiftInvar,KernelDigShiftInvar
from ..gram_matrix import FastGramMatrixLattice,FastGramMatrixDigitalNetB2
from ..pcg_module import pcg,PPCholPrecond
import numpy as np 

class FastPDEGramMatrix(_PDEGramMatrix):
    """ Fast Gram Matrix for solving PDEs 

    >>> import torch 

    >>> C1 = 1.
    >>> C2 = -1.
    >>> def u(x):
    ...     x1,x2 = x[:,0],x[:,1]
    ...     y = torch.zeros_like(x1)
    ...     r = torch.sqrt((2*x1-1)**2+(2*x2-1)**2)
    ...     b = r<1
    ...     t1 = torch.exp(-1/(1-r[b]**2))
    ...     t2 = torch.sin(torch.pi*x1[b])*torch.sin(torch.pi*x2[b])
    ...     t3 = 4*torch.sin(6*torch.pi*x1[b])*torch.sin(6*torch.pi*x2[b])
    ...     y[b] = t1*(t2+t3)
    ...     return y
    >>> def u_laplace(x):
    ...     x1g,x2g = x[:,0].clone().requires_grad_(),x[:,1].clone().requires_grad_()
    ...     xg = torch.hstack([x1g[:,None],x2g[:,None]])
    ...     yg = u(xg)
    ...     grad_outputs = torch.ones_like(yg,requires_grad=False)
    ...     yp1g = torch.autograd.grad(yg,x1g,grad_outputs,create_graph=True)[0]
    ...     yp1p1g = torch.autograd.grad(yp1g,x1g,grad_outputs,create_graph=True)[0]
    ...     yp2g = torch.autograd.grad(yg,x2g,grad_outputs,create_graph=True)[0]
    ...     yp2p2g = torch.autograd.grad(yp2g,x2g,grad_outputs,create_graph=True)[0]
    ...     return (yp1p1g+yp2p2g).detach()
    >>> def f(x):
    ...     return C1*u(x)**3+C2*u_laplace(x)
    >>> x1dticks = torch.linspace(0,1,65,dtype=float)
    >>> x1mesh,x2mesh = torch.meshgrid(x1dticks,x1dticks,indexing="ij")
    >>> x1ticks,x2ticks = x1mesh.flatten(),x2mesh.flatten()
    >>> xticks = torch.hstack([x1ticks[:,None],x2ticks[:,None]])
    >>> ymesh = u(xticks).reshape(x1mesh.shape)

    >>> ns = torch.tensor([ # number of collocation points
    ...     2**9, # on the interior
    ...     2**7, # boundary top-bottom
    ...     2**7, # boundary left-right
    ... ],dtype=int)
    >>> us = torch.tensor([ # dimensions not projected to the 0 boundary
    ...     [True,True], # interior points (not projected to the 0 boundary)
    ...     [True,False], # boundary points top-bottom i.e. x_2=0 or x_2=1
    ...     [False,True] # boundary points left-right i.e. x_1=0 or x_1=1
    ... ],dtype=bool) 
    >>> llbetas = [ # derivative orders 
    ...     [ # interior
    ...         torch.tensor([[0,0]],dtype=int), # u
    ...         torch.tensor([[2,0],[0,2]],dtype=int), # laplacian u
    ...     ],
    ...     [ # boundary top-bottom
    ...         torch.tensor([[0,0]],dtype=int) # u
    ...     ],
    ...     [ # boundary left-right 
    ...         torch.tensor([[0,0]],dtype=int) # u
    ...     ]
    ... ]
    >>> llcs = [ # summand of derivatives coefficients
    ...     [ # interior
    ...         torch.ones(1,dtype=float), # u 
    ...         torch.ones(2,dtype=float) # laplacian u
    ...     ],
    ...     [ # boundary top-bottom
    ...         torch.ones(1,dtype=float) # u
    ...     ],
    ...     [ # boundary left-right
    ...         torch.ones(1,dtype=float) # u
    ...     ]
    ... ]

    >>> def pde_lhs(ly_i, ly_b_tb, ly_b_lr):
    ...     u_i,u_laplace_i = ly_i
    ...     u_b_tb = ly_b_tb[0] 
    ...     u_b_lr = ly_b_lr[0]
    ...     lhs_i = C1*u_i**3+C2*u_laplace_i
    ...     lhs_b_tb = u_b_tb
    ...     lhs_b_lr = u_b_lr
    ...     return lhs_i,lhs_b_tb,lhs_b_lr
    >>> def pde_rhs(x_i, x_b_tb, x_b_lr):
    ...     y_i = f(x_i)
    ...     y_b_tb = torch.zeros(len(x_b_tb),dtype=float)
    ...     y_b_lr = torch.zeros(len(x_b_lr),dtype=float)
    ...     return y_i,y_b_tb,y_b_lr
        
    >>> noise = 1e-6
    >>> dd_obj = Lattice(dimension=2,seed=7) # collocation points
    >>> kernel_obj = KernelShiftInvar(dimension=2,lengthscales=1e3,scale=1,torchify=True) # kernel
    >>> ki = FastPDEGramMatrix(kernel_obj,dd_obj,ns,us,llbetas,llcs,noise) # kernel interpolant

    >>> ki._mult_check()

    >>> y,data = ki.pde_opt_gauss_newton(
    ...     pde_lhs = pde_lhs,
    ...     pde_rhs = pde_rhs)
        iter (8 max)   loss           time           
        0              2.16e+02       ...       
        1              3.41e-01       ...       
        2              3.81e-04       ...       
        3              3.53e-04       ...       
        4              3.53e-04       ...       
        5              3.53e-04       ...       
        6              3.54e-04       ...       
        7              3.54e-04       ...       
        8              3.55e-04       ...
    >>> print(data["solver"])
    CHOL

    >>> coeffs = ki._solve(y) # coeffs,rerrors_fit,times_fit = ki.pcg(y,precond=True)
    >>> kvec = ki.get_new_left_full_gram_matrix(xticks)
    >>> yhatmesh = (kvec@coeffs).reshape(x1mesh.shape)
    >>> print("L2 Rel Error ShiftInvar: %.1e"%(torch.linalg.norm(yhatmesh-ymesh)/torch.linalg.norm(ymesh)))
    L2 Rel Error ShiftInvar: 8.4e-02

    >>> dd_obj = DigitalNetB2(2,t_lms=32,alpha=2,seed=7)
    >>> kernel_obj = KernelDigShiftInvar(2,alpha=4,torchify=False)
    >>> us = np.array([
    ...     [True,True],
    ...     [True,False],
    ...     [False,True]])
    >>> ns = np.array([2**5,2**3,2**3],dtype=int)
    >>> llbetas = [
    ...     [np.array([[0,0]]),np.array([[1,0],[0,1]])],
    ...     [np.array([[0,0]])],
    ...     [np.array([[0,0]])]]
    >>> llcs = [
    ...     [np.ones(1),np.ones(2)],
    ...     [np.ones(1)],
    ...     [np.ones(1)]]
    >>> gmpde = FastPDEGramMatrix(kernel_obj,dd_obj,ns,us,llbetas,llcs)
    >>> gmpde._mult_check()
    """
    def __init__(self, kernel_obj, dd_obj, ns=None, us=None, llbetas=None, llcs=None, noise=1e-8, adaptive_noise=True, half_comp=True):
        """
        Args:
            kernel_obj (KernelShiftInvar or KernelDigShiftInvar): the kernel to use 
            dd_obj (Lattice or DigitalNetB2): the discrete distribution from which to sample points 
            ns (np.ndarray or torch.Tensor): vector of number of points on each of the regions 
            us (np.ndarray or torch.Tensor): bool matrix where each row is a region specifying the active dimensions
            llbetas (list of lists): list of length equal to the number of regions where each sub-list is the derivatives at that region 
            llcs (list of lists): list of length equal to the number of regions where each sub-list are the derivative coefficients at that region 
            noise (float): nugget term
            adaptive_noise (bool): wheather to use an adaptive nugget term
            half_comp (any): not used, just kept for consistency with PDEGramMatrix
        """
        assert isinstance(dd_obj,Lattice) or isinstance(dd_obj,DigitalNetB2)
        if us is None: us = kernel_obj.npt.ones((1,kernel_obj.d),dtype=bool)
        self.us = us 
        self.nr = len(self.us) 
        assert self.us.shape==(self.nr,kernel_obj.d) and (self.us[0]==True).all() # and (self.us.sum(1)>0).all()
        assert ns is not None, "require ns is not None"
        self.ns = ns 
        if isinstance(self.ns,int): self.ns = self.ns*kernel_obj.npt.ones(self.nr,dtype=int)
        assert self.ns.shape==(self.nr,) and (self.ns[1:]<=self.ns[0]).all()
        super(FastPDEGramMatrix,self).__init__(kernel_obj,llbetas,llcs)
        if isinstance(dd_obj,Lattice):
            gmii = FastGramMatrixLattice(kernel_obj,dd_obj,self.ns[0].item(),self.ns[0].item(),self.us[0],self.us[0],self.llbetas[0],self.llbetas[0],self.llcs[0],self.llcs[0],noise=0.,adaptive_noise=False)
        elif isinstance(dd_obj,DigitalNetB2):
            gmii = FastGramMatrixDigitalNetB2(kernel_obj,dd_obj,self.ns[0].item(),self.ns[0].item(),self.us[0],self.us[0],self.llbetas[0],self.llbetas[0],self.llcs[0],self.llcs[0],noise=0.,adaptive_noise=False)
        else:
            raise Exception("Invalid dd_obj") 
        self.gms = np.empty((self.nr,self.nr),dtype=object)
        self.gms[0,0] = gmii
        gmii__x_x = gmii._x,gmii.x 
        gmiitype = type(gmii)
        for i in range(1,self.nr):
            self.gms[0,i] = gmiitype(gmii.kernel_obj,dd_obj,gmii.n1,self.ns[i].item(),gmii.u1,self.us[i,:],gmii.lbeta1s,self.llbetas[i],gmii.lc1s_og,self.llcs[i],0.,False,gmii__x_x)
            self.gms[i,0] = gmiitype(gmii.kernel_obj,dd_obj,self.ns[i].item(),gmii.n1,self.us[i,:],gmii.u1,self.llbetas[i],gmii.lbeta1s,self.llcs[i],gmii.lc1s_og,0.,False,gmii__x_x)
            for k in range(1,self.nr):
                self.gms[i,k] = gmiitype(gmii.kernel_obj,dd_obj,self.ns[i].item(),self.ns[k].item(),self.us[i,:],self.us[k,:],self.llbetas[i],self.llbetas[k],self.llcs[i],self.llcs[k],0.,False,gmii__x_x)
        bs = [self.gms[i,0].size[0] for i in range(self.nr)]
        self.bs_cumsum = [0]+np.cumsum(bs).tolist() 
        self.length = self.bs_cumsum[-1]
        self.n_cumsum = [0]+np.cumsum(self.ns).tolist()
        self.ntot = self.n_cumsum[-1]
        self.tvec = [self.gms[i,i].t1 for i in range(self.nr)]
        self.cholesky = self.gms[0,0].cholesky
        self.cho_solve = self.gms[0,0].cho_solve
        if adaptive_noise:
            assert (self.llbetas[0][0]==0.).all() and self.llbetas[0][0].shape==(1,self.kernel_obj.d) and (self.llcs[0][0]==1.).all() and self.llcs[0][0].shape==(1,)
            full_traces = [[0. for j in range(self.tvec[i1])] for i1 in range(self.nr)]
            self.trace_ratios = [self.npt.zeros(self.tvec[i1],dtype=float) for i1 in range(self.nr)]
            for i1 in range(self.nr):
                for tt1 in range(self.tvec[i1]):
                    betas_i = self.llbetas[i1][tt1] 
                    for i2 in range(self.nr):
                        for tt2 in range(self.tvec[i2]):
                            betas_j = self.llbetas[i2][tt2]
                            if len(betas_i)==len(betas_j) and (betas_i==betas_j).all():
                                cs_i = self.llcs[i1][tt1]
                                cs_j = self.llcs[i2][tt2]
                                assert (cs_i==cs_j).all()
                                nj1 = self.gms[i2,i2].n1
                                full_traces[i1][tt1] += nj1*self.gms[i2,i2].k00diags[tt2]
                    self.trace_ratios[i1][tt1] = full_traces[i1][tt1]/full_traces[0][0]
            for i in range(self.nr):
                for tt in range(self.tvec[i]):
                    self.gms[i,i].lam[tt,tt][0,0,0,:] += noise*self.trace_ratios[i][tt]
        else: 
            assert all(self.gms[i,i].invertible for i in range(self.nr))
            for i in range(self.nr):
                for tt1 in range(self.tvec[i]):
                    self.gms[i,i].lam[tt1,tt1][0,0,0,:] += noise
        self._full_mat = None
        self.___xs = None
        self.__xs = None
    @property
    def full_mat(self):
        if self._full_mat is None: 
            self._full_mat = self@self.npt.eye(self.length,dtype=float)
        return self._full_mat
    @property
    def _xs(self):
        if self.___xs is None:
            self.___xs = [None]*self.nr
            for i in range(self.nr):
                self.___xs[i] = self.gms[i,0].clone(self.gms[i,0]._x1)
        return self.___xs
    @property
    def xs(self):
        if self.__xs is None:
            self.__xs = [self.gms[i,0]._convert__x_to_x(_x) for i,_x in enumerate(self._xs)]
        return self.__xs
    def __matmul__(self, y):
        yogndim = y.ndim
        assert yogndim<=2 
        if yogndim==1: y = y[:,None] # (l,v)
        assert y.ndim==2 and y.shape[0]==(self.length)
        ysplit = np.split(y,self.bs_cumsum[1:-1]) # (li,v)
        ssplit = [0. for i in range(self.nr)]
        for i in range(self.nr):
            for k in range(self.nr):
                ssplit[i] += self.gms[i,k]@ysplit[k]
        s = self.npt.vstack(ssplit) 
        return s[:,0] if yogndim==1 else s
    
      