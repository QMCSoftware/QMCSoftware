import numpy as np 
import lightning
import torch 
import gpytorch

class LMOpLearnLowerTriMatGP(lightning.LightningModule):
    def __init__(self, gp, automatic_optimization=True, input_relaxation=False, fixed_noise=True, learning_rate=1.):
        super().__init__()
        self.gp = gp
        self.automatic_optimization = automatic_optimization
        self.input_relaxation = input_relaxation
        self.pred_inv = True
        self.ftype = gp.variational_strategy.base_variational_strategy.inducing_points.dtype
        self.s0 = gp.variational_strategy.base_variational_strategy.inducing_points.size(-1)
        k = gp.num_tasks
        self.n = int(1/2*(np.sqrt(8*k+1)-1))
        self._tril_i0,self._tril_i1 = torch.tril_indices(self.n,self.n)
        self._diag_i = torch.arange(self.n)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.gp.num_tasks,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
            has_task_noise = False,
            has_global_noise = True)
        self.mll = gpytorch.mlls.VariationalELBO(likelihood,self.gp,num_data=self.s0)
        if fixed_noise:
            self.mll.likelihood.raw_noise = torch.nn.Parameter(-torch.inf*torch.ones_like(self.mll.likelihood.raw_noise),requires_grad=False)
        self.lr = learning_rate
    def to(self, device):
        super().to(device)
        self.gp = self.gp.to(device)
        self.mll.likelihood = self.mll.likelihood.to(device)
        self._tril_i0,self._tril_i1,self._diag_i = self._tril_i0.to(device),self._tril_i1.to(device),self._diag_i.to(device)
    def forward_mvn(self, v, relaxations=None):
        assert v.ndim==2
        ins = v[:,:self.s0]
        if self.input_relaxation:
            assert relaxations is not None
            ins = torch.hstack([ins,relaxations[:,None]])
        return self.gp(ins)
    def forward(self, v, relaxations=None):
        outs = self.forward_mvn(v,relaxations)
        L_chol_hat = torch.zeros((v.size(0),self.n,self.n),dtype=self.ftype,device=self._tril_i0.device)
        L_chol_hat[:,self._tril_i0,self._tril_i1] = outs.mean
        return L_chol_hat
    def linsolve(self, y, v, relaxations=None, inference_mode=True, return_L_hat=False):
        # v should be (R,n) and y should be (R,N,K) to give L_hat which is (R,N,N) and x which is (R,N,K)
        assert y.dtype==v.dtype
        dtype = v.dtype
        yis1d = y.ndim==1
        if yis1d:
            y = y[:,None]
        assert v.ndim in [1,2] and (v.ndim+1)==y.ndim
        vis1d = v.ndim==1
        if v.ndim==1:
            v = v[None,:]
            y = y[None,:,:]
        assert v.size(0)==y.size(0)
        v,y = v.to(self.ftype),y.to(self.ftype)
        if inference_mode:
            with torch.inference_mode():
                L_hat = self.forward(v,relaxations)
        else:
            L_hat = self.forward(v)
        if self.pred_inv:
            L_hat_T = torch.transpose(L_hat,-2,-1)
            x = torch.bmm(L_hat_T,torch.bmm(L_hat,y))
        else: 
           x = torch.cholesky_solve(y,L_hat)
        x = x.to(dtype)
        x = x[0] if vis1d else x
        x = x[:,0] if yis1d else x
        if not return_L_hat:
            return x 
        else:
            L_hat = L_hat[0] if vis1d else L_hat 
            return x,L_hat
    def _common_step(self, batch, tag):
        v,relaxations,L_chol,L_inv_chol = batch
        out = self.forward_mvn(v,relaxations)
        loss = -self.mll(out,L_inv_chol[:,self._tril_i0,self._tril_i1])
        fnorm = torch.sqrt(torch.mean((out.mean-L_inv_chol[:,self._tril_i0,self._tril_i1])**2))
        self.log(tag+"_loss",loss,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        self.log(tag+"_fnorm",fnorm,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        return loss
    def training_step(self, batch, batch_idx):
        self.gp.train()
        self.mll.likelihood.train()
        if self.automatic_optimization:
            return self._common_step(batch,tag="train")
        else:
            opt = self.optimizers()
            def closure():
                loss = self._common_step(batch,tag="train")
                opt.zero_grad(set_to_none=True)
                self.manual_backward(loss)
                return loss
            opt.step(closure=closure)
    def validation_step(self, batch, batch_idx):
        self.gp.eval()
        self.mll.likelihood.eval()
        with torch.inference_mode(True):
          return self._common_step(batch,tag="val")
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),amsgrad=True,lr=self.lr)
        # if self.trainer.current_epoch>=2000:
        #     optimizer = torch.optim.LBFGS(self.parameters())#,lr=1e-2,amsgrad=True)
        return optimizer