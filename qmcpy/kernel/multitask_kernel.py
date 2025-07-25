from .abstract_kernel import AbstractKernel
from .common_kernels import KernelGaussian
from ..util.transforms import tf_identity,tf_exp_eps,tf_exp_eps_inv,insert_batch_dims
import numpy as np 


class KernelMultiTask(AbstractKernel):

    r"""
    Multi-task kernel
    
    $$K((i,\boldsymbol{x}),(j,\boldsymbol{z})) = K_{\mathrm{task}}(i,j) K_{\mathrm{base}}(\boldsymbol{x},\boldsymbol{z})$$
    
    parameterized for $T$ tasks by a factor $\mathsf{F} \in \mathbb{R}^{T \times r}$ and a diagonal $\boldsymbol{v} \in \mathbb{R}^T$ so that 
    
    $$\left[K_{\mathrm{task}}(i,j)\right]_{i,j=1}^T = \mathsf{F} \mathsf{F}^T + \mathrm{diag}(\boldsymbol{v}).$$ 
    
    Examples:
        >>> kmt = KernelMultiTask(KernelGaussian(d=2),num_tasks=3,diag=[1,2,3])
        >>> x = np.random.rand(4,2) 
        >>> task = np.arange(3) 
        >>> kmt(x[0],x[0],task[1],task[1]).item()
        3.0
        >>> kmt(x[0],x[0],task,task)
        array([2., 3., 4.])
        >>> kmt(x[0],x[0],task[:,None],task[None,:]).shape
        (3, 3)
        >>> kmt(x,x,task[1],task[1])
        array([3., 3., 3., 3.])
        >>> kmt(x[:3],x[:3],task,task)
        array([2., 3., 4.])
        >>> kmt(x,x,task[:,None],task[:,None]).shape
        (3, 4)
        >>> v = kmt(x[:3,None,:],x[None,:3,:],task,task)
        >>> v.shape
        (3, 3)
        >>> np.allclose(v,kmt.base_kernel(x[:3,None,:],x[None,:3,:])*kmt.taskmat[task,task])
        True
        >>> kmt(x[:,None,:],x[None,:,:],task[:,None,None],task[:,None,None]).shape
        (3, 4, 4)
        >>> kmt(x[:,None,:],x[None,:,:],task[:,None,None,None],task[None,:,None,None]).shape
        (3, 3, 4, 4)

        Batched inference

        >>> kernel_base = KernelGaussian(d=10,shape_lengthscales=(5,1),shape_scale=(3,5,1))
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kmt = KernelMultiTask(kernel_base,
        ...     num_tasks=4,
        ...     shape_factor=(6,3,5,4,2),
        ...     diag = rng.uniform(low=0,high=1,size=(3,5,4)))
        >>> x = np.random.rand(8,10)
        >>> task = np.arange(4) 
        >>> kmt(x[0],x[0],task[0],task[0]).shape
        (6, 3, 5)
        >>> kmt(x[0],x[0],task,task).shape
        (6, 3, 5, 4)
        >>> kmt(x,x,task[0],task[0]).shape
        (6, 3, 5, 8)
        >>> v = kmt(x[:4,None,:],x[None,:4,:],task,task)
        >>> v.shape
        (6, 3, 5, 4, 4)
        >>> kmat_x = kmt.base_kernel(x[:4,None,:],x[None,:4,:])
        >>> kmat_x.shape
        (3, 5, 4, 4)
        >>> kmat_tasks = kmt.taskmat[...,task,task]
        >>> kmat_tasks.shape
        (6, 3, 5, 4)
        >>> np.allclose(v,kmat_tasks[...,None,:]*kmat_x)
        True
        >>> np.allclose(v,kmat_tasks[...,:,None]*kmat_x)
        False
        >>> kmt(x[:,None,:],x[None,:,:],task[:,None,None,None],task[None,:,None,None]).shape
        (6, 3, 5, 4, 4, 8, 8)
        >>> kmt(x[:,None,None,None,:],x[None,:,None,None,:],task[:,None],task[None,:]).shape
        (6, 3, 5, 8, 8, 4, 4)

        PyTorch
        
        >>> import torch 
        >>> kmt = KernelMultiTask(KernelGaussian(d=2,torchify=True),num_tasks=3,diag=[1,2,3])
        >>> x = torch.from_numpy(np.random.rand(4,2))
        >>> task = np.arange(3) 
        >>> kmt(x[0],x[0],task[1],task[1]).item()
        3.0
        >>> kmt(x[0],x[0],task,task)
        tensor([2., 3., 4.], dtype=torch.float64, grad_fn=<SelectBackward0>)
        >>> kmt(x[0],x[0],task[:,None],task[None,:]).shape
        torch.Size([3, 3])
        >>> kmt(x,x,task[1],task[1])
        tensor([3., 3., 3., 3.], dtype=torch.float64, grad_fn=<SelectBackward0>)
        >>> kmt(x[:3],x[:3],task,task)
        tensor([2., 3., 4.], dtype=torch.float64, grad_fn=<SelectBackward0>)
        >>> kmt(x,x,task[:,None],task[:,None]).shape
        torch.Size([3, 4])
        >>> v = kmt(x[:3,None,:],x[None,:3,:],task,task)
        >>> v.shape
        torch.Size([3, 3])
        >>> torch.allclose(v,kmt.base_kernel(x[:3,None,:],x[None,:3,:])*kmt.taskmat[task,task])
        True
        >>> kmt(x[:,None,:],x[None,:,:],task[:,None,None],task[:,None,None]).shape
        torch.Size([3, 4, 4])
        >>> kmt(x[:,None,:],x[None,:,:],task[:,None,None,None],task[None,:,None,None]).shape
        torch.Size([3, 3, 4, 4])

        Batched inference

        >>> kernel_base = KernelGaussian(d=10,shape_lengthscales=(5,1),shape_scale=(3,5,1),torchify=True)
        >>> rng = np.random.Generator(np.random.PCG64(7))
        >>> kmt = KernelMultiTask(kernel_base,
        ...     num_tasks=4,
        ...     shape_factor=(6,3,5,4,2),
        ...     diag = rng.uniform(low=0,high=1,size=(3,5,4)))
        >>> x = torch.from_numpy(np.random.rand(8,10))
        >>> task = torch.arange(4) 
        >>> kmt(x[0],x[0],task[0],task[0]).shape
        torch.Size([6, 3, 5])
        >>> kmt(x[0],x[0],task,task).shape
        torch.Size([6, 3, 5, 4])
        >>> kmt(x,x,task[0],task[0]).shape
        torch.Size([6, 3, 5, 8])
        >>> v = kmt(x[:4,None,:],x[None,:4,:],task,task)
        >>> v.shape
        torch.Size([6, 3, 5, 4, 4])
        >>> kmat_x = kmt.base_kernel(x[:4,None,:],x[None,:4,:])
        >>> kmat_x.shape
        torch.Size([3, 5, 4, 4])
        >>> kmat_tasks = kmt.taskmat[...,task,task]
        >>> kmat_tasks.shape
        torch.Size([6, 3, 5, 4])
        >>> torch.allclose(v,kmat_tasks[...,None,:]*kmat_x)
        True
        >>> torch.allclose(v,kmat_tasks[...,:,None]*kmat_x)
        False
        >>> kmt(x[:,None,:],x[None,:,:],task[:,None,None,None],task[None,:,None,None]).shape
        torch.Size([6, 3, 5, 4, 4, 8, 8])
        >>> kmt(x[:,None,None,None,:],x[None,:,None,None,:],task[:,None],task[None,:]).shape
        torch.Size([6, 3, 5, 8, 8, 4, 4])
        >>> kmt.factor.dtype
        torch.float32
        >>> kmt.diag.dtype
        torch.float64
    """

    def __init__(self,
            base_kernel,
            num_tasks, 
            factor = 1.,
            diag =  1.,
            shape_factor = None, 
            shape_diag = None,
            tfs_factor = (tf_identity,tf_identity),
            tfs_diag = (tf_exp_eps_inv,tf_exp_eps),
            requires_grad_factor = True, 
            requires_grad_diag = True,
            rank_factor = 1
            ):
        r"""
        Args:
            base_kernel (AbstractKernel): $K_{\mathrm{base}}$. 
            num_tasks (int): Number of tasks $T>1$. 
            factor (Union[np.ndarray,torch.Tensor]): Factor $\mathsf{F}$. 
            diag (Union[np.ndarray,torch.Tensor]): Diagonal parameter $\boldsymbol{v}$. 
            shape_factor (list): Shape of `factor` when `np.isscalar(factor)`. 
            shape_diag (list): Shape of `diag` when `np.isscalar(diag)`. 
            tfs_factor (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            tfs_diag (Tuple[callable,callable]): The first argument transforms to the raw value to be optimized; the second applies the inverse transform.
            requires_grad_factor (bool): If `True` and `torchify`, set `requires_grad=True` for `factor`.
            requires_grad_diag (bool): If `True` and `torchify`, set `requires_grad=True` for `diag`.
        """
        assert isinstance(base_kernel,AbstractKernel)
        super().__init__(
            d = base_kernel.d,
            torchify = base_kernel.torchify,
            device = base_kernel.device,
            compile_call = False,
            comiple_call_kwargs = {})
        self.base_kernel = base_kernel
        self.AUTOGRADKERNEL = base_kernel.AUTOGRADKERNEL 
        assert np.isscalar(num_tasks) and num_tasks%1==0 and num_tasks>1
        self.num_tasks = num_tasks
        assert np.isscalar(rank_factor) and rank_factor%1==0 and 1<=rank_factor<=self.num_tasks
        self.raw_factor,self.tf_factor = self.parse_assign_param(
            pname = "factor",
            param = factor,
            shape_param = [self.num_tasks,rank_factor] if shape_factor is None else shape_factor,
            requires_grad_param = requires_grad_factor,
            tfs_param = tfs_factor,
            endsize_ops = list(range(self.num_tasks+1)),
            constraints = [])
        assert self.raw_factor.shape[-2]==self.num_tasks
        self.raw_diag,self.tf_diag = self.parse_assign_param(
            pname = "diag",
            param = diag, 
            shape_param = [self.num_tasks] if shape_diag is None else shape_diag,
            requires_grad_param = requires_grad_diag,
            tfs_param = tfs_diag,
            endsize_ops = [1,self.num_tasks],
            constraints = ["POSITIVE"])
        self.eye_num_tasks = self.npt.eye(self.num_tasks,**self.nptkwargs)
        self.batch_params["taskmat"] = self.taskmat
        self.nbdim_base = self.base_kernel.nbdim
    
    @property
    def factor(self):
        return self.tf_factor(self.raw_factor)
    
    @property
    def diag(self):
        return self.tf_diag(self.raw_diag)

    @property
    def taskmat(self):
        factor = self.factor
        diag = self.diag
        taskmat = self.npt.einsum("...ij,...kj->...ik",factor,factor)+diag[...,None]*self.eye_num_tasks
        return taskmat
    
    def __call__(self, x0, x1, task0, task1, beta0=None, beta1=None, c=None):
        r"""
        Evaluate the kernel with (optional) partial derivatives 

        $$\sum_{\ell=1}^p c_\ell \partial_{\boldsymbol{x}_0}^{\boldsymbol{\beta}_{\ell,0}} \partial_{\boldsymbol{x}_1}^{\boldsymbol{\beta}_{\ell,1}} K((\boldsymbol{x}_0,i_0),(\boldsymbol{x}_1,i_1)).$$
        
        Args:
            x0 (Union[np.ndarray,torch.Tensor]): Shape `x0.shape=(...,d)` first input to kernel.
            x1 (Union[np.ndarray,torch.Tensor]): Shape `x1.shape=(...,d)` second input to kernel. 
            task0 (Union[int,np.ndarray,torch.Tensor]): First task indices $i_0$. 
            task1 (Union[int,np.ndarray,torch.Tensor]): Second task indices $i_1$.
            beta0 (Union[np.ndarray,torch.Tensor]): Shape `beta0.shape=(p,d)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_0$.
            beta1 (Union[np.ndarray,torch.Tensor]): Shape `beta1.shape=(p,d)` derivative orders with respect to first inputs, $\boldsymbol{\beta}_1$.
            c (Union[np.ndarray,torch.Tensor]): Shape `c.shape=(p,)` coefficients of derivatives.

        Returns:
            k (Union[np.ndarray,torch.Tensor]): Kernel evaluations with batched shape, see the doctests for examples. 
        """
        kmat_x = self.base_kernel.__call__(x0,x1,beta0,beta1,c)[...,None]
        nnbdim_x = kmat_x.ndim-self.nbdim_base-1
        kmat_tasks = self.taskmat
        nbdim = kmat_tasks.ndim-2
        kmat_tasks = kmat_tasks[...,task0,task1,None]
        nnbdim_kmat = kmat_tasks.ndim-nbdim-1
        commondim = max(nnbdim_x,nnbdim_kmat)
        kmat_x = insert_batch_dims(kmat_x,commondim-nnbdim_x,self.nbdim_base)
        kmat_tasks = insert_batch_dims(kmat_tasks,commondim-nnbdim_kmat,nbdim)
        kmat = kmat_x*kmat_tasks 
        return kmat[...,0]
