import lightning
import torch
import gpytorch

class LMOpLearnClassicGP(lightning.LightningModule):
    def __init__(self, gp, automatic_optimization=True, fixed_noise=True, learning_rate=1., noise_lb=1e-8):
        super().__init__()
        self.gp = gp
        self.automatic_optimization = automatic_optimization
        self.ftype = gp.variational_strategy.base_variational_strategy.inducing_points.dtype
        self.s0 = gp.variational_strategy.base_variational_strategy.inducing_points.size(-1)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.gp.num_tasks,
            noise_constraint=gpytorch.constraints.GreaterThan(noise_lb),
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
    def forward_mvn(self, u):
        assert u.ndim==2
        return self.gp(u)
    def forward(self, u):
        outs = self.forward_mvn(u)
        return outs.mean
    def _common_step(self, batch, tag):
        u,v = batch
        vhat_mvn = self.forward_mvn(u)
        loss = -self.mll(vhat_mvn,v)
        vhat = vhat_mvn.mean
        mse = torch.mean(torch.sqrt(torch.mean((vhat-v)**2)))
        self.log(tag+"_loss",loss,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        self.log(tag+"_mse",mse,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
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