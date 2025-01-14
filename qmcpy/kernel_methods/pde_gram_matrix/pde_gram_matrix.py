from ._pde_gram_matrix import _PDEGramMatrix
from ...discrete_distribution import IIDStdUniform
from ...discrete_distribution._discrete_distribution import DiscreteDistribution
from ...discrete_distribution import Lattice
from ..kernel import KernelGaussian
from ..gram_matrix import GramMatrix
from ..pcg_module import pcg,PPCholPrecond
import numpy as np 
import itertools

class PDEGramMatrix(_PDEGramMatrix):
    """ Gram Matrix for solving PDEs 
    
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
    >>> lattice = Lattice(dimension=2,seed=7) # collocation points
    >>> kernel = KernelGaussian(dimension=2,lengthscales=5e-2,scale=1.,torchify=True) # kernel
    >>> ki = PDEGramMatrix(kernel,lattice,ns,us,llbetas,llcs,noise) # kernel interpolant
    
    >>> ki._mult_check()

    >>> y,data = ki.pde_opt_gauss_newton(
    ...     pde_lhs = pde_lhs,
    ...     pde_rhs = pde_rhs)
        iter (8 max)   loss           
        0              2.16e+02       
        1              3.88e-01       
        2              2.58e-04       
        3              1.11e-07       
        4              5.05e-08       
        5              4.18e-08       
        6              4.52e-08       
        7              4.95e-08       
        8              4.44e-08       
    >>> print(data["solver"])
    CHOL

    >>> coeffs = ki._solve(y) # coeffs,rerrors_fit,times_fit = ki.pcg(y,precond=True)
    >>> kvec = ki.get_new_left_full_gram_matrix(xticks)
    >>> yhatmesh = (kvec@coeffs).reshape(x1mesh.shape)
    >>> print("L2 Rel Error Gauss: %.1e"%(torch.linalg.norm(yhatmesh-ymesh)/torch.linalg.norm(ymesh)))
    L2 Rel Error Gauss: 5.8e-02

    >>> y,data = ki.pde_opt_gauss_newton(
    ...     pde_lhs = pde_lhs,
    ...     pde_rhs = pde_rhs,
    ...     precond_setter = lambda pde_gm: PPCholPrecond(pde_gm.rtheta.full_mat,rtol=1e-7),
    ...     pcg_kwargs = {"rtol":1e-3},
    ...     verbose = 2)
        iter (8 max)   loss           K(A)           K(P)           K(P)/K(A)      Lk.shape       PCG rberror    PCG steps (768 max)
        0              2.16e+02       
        1              4.23e-01       1.5e+13        2.4e+08        1.6e-05        (768, 495)     9.2e-04        286
        2              2.77e-01       1.5e+13        3.0e+08        2.0e-05        (768, 526)     1.3e-03        768
        3              2.11e-01       1.5e+13        3.0e+08        2.0e-05        (768, 522)     9.8e-04        505
        4              1.85e-01       1.5e+13        3.0e+08        2.0e-05        (768, 526)     8.5e-04        534
        5              2.12e-01       1.5e+13        3.0e+08        2.0e-05        (768, 522)     9.8e-04        484
        6              2.15e-01       1.5e+13        3.0e+08        2.0e-05        (768, 522)     1.0e-03        525
        7              2.06e-01       1.5e+13        3.3e+08        2.2e-05        (768, 526)     9.5e-04        508
        8              2.12e-01       1.5e+13        3.1e+08        2.0e-05        (768, 526)     9.8e-04        530
    >>> print(data["solver"])
    PCG-PPCholPrecond

    >>> precond = PPCholPrecond(ki.full_mat)
    >>> print(precond.info_str(ki.full_mat))
    K(A)           K(P)           K(P)/K(A)      Lk.shape       
    2.8e+13        8.4e+04        3.0e-09        (1280, 695)    
    >>> coeffs,data = pcg(ki,y,precond,rtol=5e-3)
    >>> kvec = ki.get_new_left_full_gram_matrix(xticks)
    >>> yhatmesh = (kvec@coeffs).reshape(x1mesh.shape)
    >>> print("L2 Rel Error Gauss: %.1e"%(torch.linalg.norm(yhatmesh-ymesh)/torch.linalg.norm(ymesh)))
    L2 Rel Error Gauss: 7.1e-02
    """
    def __init__(self, kernel_obj, xs, ns=None, us=None, llbetas=None, llcs=None, noise=1e-8, adaptive_noise=True, half_comp=True):
        """
        Args:
            kernel_obj (_Kernel): the kernel to use
            xs (DiscreteDisribution or list of (numpy.ndarray or torch.Tensor)): locations at which the regions are sampled 
            ns (np.ndarray or torch.Tensor): vector of number of points on each of the regions. 
                Only used when a DiscreteDistribution is passed in for xs
            us (np.ndarray or torch.Tensor): bool matrix where each row is a region specifying the active dimensions
                Only used when a DiscreteDistribution is passed in for xs
            llbetas (list of lists): list of length equal to the number of regions where each sub-list is the derivatives at that region 
            llcs (list of lists): list of length equal to the number of regions where each sub-list are the derivative coefficients at that region 
            noise (float): nugget term
            adaptive_noise (bool): wheather to use an adaptive nugget term
            half_comp (bool or numpy.ndarray or torch.Tensor): if True, put project half the points to the 0 boundary and half the points to the 1 boundary
        """
        if isinstance(xs,DiscreteDistribution):
            dd_obj = xs
            assert ns is not None and ns.ndim==1 and (ns[1:]<=ns[0]).all()
            nr = len(ns)
            assert us is not None and us.ndim==2 and us.shape==(nr,kernel_obj.d) and (us[0]==True).all() # and (us.sum(1)>0).all()
            x = dd_obj.gen_samples(ns[0])
            if kernel_obj.torchify: 
                import torch 
                x = torch.from_numpy(x)
                xs = [x[:n].clone() for n in ns]
            else: # numpyify 
                xs = [x[:n].copy() for n in ns]
            if isinstance(half_comp,bool): half_comp = half_comp*torch.ones(len(us),dtype=bool)
            assert len(half_comp)==len(us)
            for i,u in enumerate(us):
                if half_comp[i]:
                    nhalf = ns[i]//2
                    xs[i][:nhalf,~u] = 1. 
                    xs[i][nhalf:,~u] = 0.
                else:
                    xs[i][:,~u] = 0.
        assert isinstance(xs,list)
        self.xs = xs 
        self.nr = len(self.xs)
        self.ns = np.array([len(x) for x in self.xs],dtype=int)
        super(PDEGramMatrix,self).__init__(kernel_obj,llbetas,llcs)
        self.gms = np.empty((self.nr,self.nr),dtype=object)
        for i1,i2 in itertools.product(range(self.nr),range(self.nr)):
            self.gms[i1,i2] = GramMatrix(self.kernel_obj,self.xs[i1],self.xs[i2],self.llbetas[i1],self.llbetas[i2],self.llcs[i1],self.llcs[i2],noise=0.,adaptive_noise=False)
        bs = [self.gms[i,0].size[0] for i in range(self.nr)]
        self.bs_cumsum = [0]+np.cumsum(bs).tolist() 
        self.length = self.bs_cumsum[-1]
        self.n_cumsum = [0]+np.cumsum(self.ns).tolist()
        self.ntot = self.n_cumsum[-1]
        self.tvec = [self.gms[i,i].t1 for i in range(self.nr)]
        if adaptive_noise:
            assert (self.llbetas[0][0]==0.).all() and self.llbetas[0][0].shape==(1,self.kernel_obj.d) and (self.llcs[0][0]==1.).all() and self.llcs[0][0].shape==(1,)
            full_traces = [[0. for j in range(self.tvec[i1])] for i1 in range(self.nr)]
            self.trace_ratios = [self.npt.zeros(self.tvec[i1]) for i1 in range(self.nr)]
            for i1 in range(self.nr):
                ni1 = self.gms[i1,i1].n1
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
                                full_traces[i1][tt1] += nj1*self.gms[i2,i2].full_mat[tt2*nj1,tt2*nj1]
                    self.trace_ratios[i1][tt1] = full_traces[i1][tt1]/full_traces[0][0]
            for i in range(self.nr):
                ni1 = self.gms[i,i].n1
                self.gms[i,i].full_mat += noise*self.npt.diag((self.npt.ones((ni1,self.tvec[i]))*self.trace_ratios[i]).T.flatten())
                self.full_mat = self.npt.vstack([self.npt.hstack([self.gms[i,k].full_mat for k in range(self.nr)]) for i in range(self.nr)])
        else:
            assert all(self.gms[i,i].invertible for i in range(self.nr))
            self.full_mat = self.npt.vstack([self.npt.hstack([self.gms[i,k].full_mat for k in range(self.nr)]) for i in range(self.nr)])
            self.full_mat += noise*self.npt.eye(self.length,dtype=float,**self.ckwargs)
        self.cholesky = self.gms[0,0].cholesky
        self.cho_solve = self.gms[0,0].cho_solve
    def __matmul__(self, y):
        return self.full_mat@y
    def get_xs(self):
        return [self.gms[0,0].clone(x) for x in self.xs]
    
    
