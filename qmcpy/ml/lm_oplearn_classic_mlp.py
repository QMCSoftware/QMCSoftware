import lightning 
import torch 

class LMOpLearnClassicMLP(lightning.LightningModule):
    def __init__(self, mlp, automatic_optimization=True, learning_rate=1e-3):
        super().__init__()
        self.mlp = mlp
        self.automatic_optimization = automatic_optimization
        self.ftype = self.mlp.mlp_sequential[0].weight.dtype
        self.s0 = self.mlp.mlp_sequential[0].weight.size(1)
        self.lr = learning_rate
    def to(self, device):
        super().to(device)
        self.mlp = self.mlp.to(device)
    def forward(self, u):
        assert u.ndim==2
        return self.mlp(u)
    def _common_step(self, batch, tag):
        u,v = batch
        vhat = self.forward(u)
        mse = torch.mean(torch.sqrt(torch.mean((vhat-v)**2)))
        self.log(tag+"_mse",mse,logger=True,sync_dist=True,on_step=False,on_epoch=True,prog_bar=True)
        return mse
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