import numpy as np 
import lightning
import torch 

class LMOpLearnLowerTriMatMLP(lightning.LightningModule):
    def __init__(self, mlp, automatic_optimization=True, input_relaxation=False, pred_inv=True, fnorm_weight=1., kl_weight=1., learning_rate=1e-3):
        super().__init__()
        self.mlp = mlp
        self.automatic_optimization = automatic_optimization
        self.input_relaxation = input_relaxation
        self.pred_inv = pred_inv
        self.ftype = self.mlp.mlp_sequential[0].weight.dtype
        self.s0 = self.mlp.mlp_sequential[0].weight.size(1)
        k = self.mlp.mlp_sequential[-1].weight.size(0)
        self.n = int(1/2*(np.sqrt(8*k+1)-1))
        self._tril_i0,self._tril_i1 = torch.tril_indices(self.n,self.n)
        self._diag_i = torch.arange(self.n)
        self.fnorm_weight = fnorm_weight
        self.kl_weight = kl_weight
        self.lr = learning_rate
        assert self.fnorm_weight!=0 or self.kl_weight!=0
    def to(self, device):
        super().to(device)
        self.mlp = self.mlp.to(device)
        self._tril_i0 = self._tril_i0.to(device)
        self._tril_i1 = self._tril_i1.to(device)
        self._diag_i = self._diag_i.to(device)
    def forward(self, v, relaxations=None):
        assert v.ndim==2
        ins = v[:,:self.s0]
        if self.input_relaxation:
            assert relaxations is not None
            ins = torch.hstack([ins,relaxations[:,None]])
        L_chol_hat = torch.zeros((v.size(0),self.n,self.n),dtype=self.ftype,device=self._tril_i0.device)
        L_chol_hat[:,self._tril_i0,self._tril_i1] = self.mlp(ins)
        # set diagonal to exp(x-1) if x<1 and leave as x otherwise
        diags = L_chol_hat[:,self._diag_i,self._diag_i]
        diags[diags<1] = torch.exp(diags[diags<1]-1)
        L_chol_hat[:,self._diag_i,self._diag_i] = diags
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
    def frob_norm(self, L1, L2):
        ogndim = L1.ndim
        assert ogndim in [2,3]
        if ogndim==2:
            L1 = L1[None,:,:]
            L2 = L2[None,:,:]
        assert L1.ndim==3 and L2.ndim==3
        diff = L1[:,self._tril_i0,self._tril_i1]-L2[:,self._tril_i0,self._tril_i1]
        fnorms = torch.mean(diff**2,dim=1)
        return fnorms[0] if ogndim==2 else fnorms 
    def kl_div(self, L1, L2): # eq (C.1) in https://arxiv.org/pdf/2304.01294
        ogndim = L1.ndim
        assert ogndim in [2,3]
        if ogndim==2:
            L1 = L1[None,:,:]
            L2 = L2[None,:,:]
        assert L1.ndim==3 and L1.ndim==3
        det2_1 = 2*torch.log(L1.diagonal(dim1=-1,dim2=-2)).sum(-1)
        det2_2 = 2*torch.log(L2.diagonal(dim1=-1,dim2=-2)).sum(-1)
        L1T = torch.transpose(L1,dim0=-2,dim1=-1)
        L2T = torch.transpose(L2,dim0=-2,dim1=-1)
        traces = (torch.bmm(L2,L1)**2).sum(1).sum(1)
        kls = 1/2*(-det2_1-det2_2+traces-L1.size(-1))
        return kls
    def _common_step(self, batch, tag):
        v,relaxations,L_chol,L_inv_chol = batch
        L_hat = self.forward(v,relaxations)
        #if self.loss_metric=="FROB": # mean (Frobineous norm)
        fnorm = torch.nan if self.fnorm_weight==0 else torch.sqrt(torch.mean(self.frob_norm(L_inv_chol,L_hat) if self.pred_inv else self.frob_norm(L_chol,L_hat)))
        klnorm = torch.nan if self.kl_weight==0 else torch.mean(self.kl_div(L_chol,L_hat) if self.pred_inv else self.kl_div(L_hat,L_inv_chol))
        loss = (0 if self.fnorm_weight==0 else self.fnorm_weight*fnorm)+(0 if self.kl_weight==0 else self.kl_weight*klnorm)
        self.log(tag+"_fnorm",fnorm,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        self.log(tag+"_kl",klnorm,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        self.log(tag+"_loss",loss,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        return loss
    def training_step(self, batch, batch_idx):
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
        with torch.inference_mode(True):
          return self._common_step(batch,tag="val")
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr,amsgrad=True)
        # if self.trainer.current_epoch>=2000:
        #     optimizer = torch.optim.LBFGS(self.parameters())#,lr=1e-2,amsgrad=True)
        return optimizer