from .abstract_kernel import AbstractKernel
from ..util.transforms import tf_identity,tf_exp_eps,tf_exp_eps_inv
import numpy as np 


class KernelMultiTask(AbstractKernel):

    r"""
    Multi-task kernel
    
    $$K((i,\boldsymbol{x}),(j,\boldsymbol{z})) = K_{\mathrm{task})(i,j) K_{\mathrm{base}}(\boldsymbol{x},\boldsymbol{z})
    
    parameterized for $T$ tasks by a factor $\mathsf{F} \in \mathbb{R}^{T \times r}$ and a diagonal $\boldsymbol{v} \in \mathbb{R}^T$ so that 
    
    $$\left[K_{\mathrm{task}}(i,j)\right]_{i,j=1}^T = \mathsf{F} \mathsf{F}^T + \mathrm{diag}(\boldsymbol{v}).$$ 
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
        super().__init__(d=base_kernel.d,shape_batch=base_kernel.shape_batch,torchify=base_kernel.torchify,device=base_kernel.device)
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
        self.batch_params["factor"] = self.factor
        self.raw_diag,self.tf_diag = self.parse_assign_param(
            pname = "diag",
            param = diag, 
            shape_param = [self.num_tasks] if shape_diag is None else shape_diag,
            requires_grad_param = requires_grad_diag,
            tfs_param = tfs_diag,
            endsize_ops = [1,self.num_tasks],
            constraints = ["POSITIVE"])
        self.batch_params["diag"] = self.diag
    
    @property
    def factor(self):
        return self.tf_factor(self.raw_factor)
    
    @property
    def diag(self):
        return self.tf_diag(self.raw_diag)
    
    def __call__(self, *args, **kwargs):
        assert False, "not implemented"
