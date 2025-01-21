import torch 

class DatasetClassicOpLearn(torch.utils.data.Dataset):
    def __init__(self, u, v, device="cpu", fp32=False):
        self.u = u 
        self.v = v
        assert self.u.size(0)==self.v.size(0)
        if fp32:
            self.u = self.u.to(torch.float32) 
            self.v = self.v.to(torch.float32)
        self.i = torch.arange(self.u.size(0))
        self._put_on_device(device)
    def _put_on_device(self, device):
        self.i = self.i.to(device)
        self.u = self.u.to(device)
        self.v = self.v.to(device)
    def __getitems__(self, i):
        i = torch.tensor(i,dtype=torch.int)
        i = self.i[i]
        return self.u[i],self.v[i]
    def __len__(self):
        return len(self.i)

class DatasetLowerTriMatOpLearn(torch.utils.data.Dataset):
    def __init__(self, v, L_chols, relaxations=torch.zeros(1), device="cpu", steps="all", fp32=False):
        self.v = v 
        self.relaxations = relaxations
        self.L_chols = L_chols
        common_v0 = (self.L_chols[0,0,:,:]==self.L_chols[:,0,:,:]).all()
        if (relaxations==0).all():
            self.L_chols = L_chols[:,:,None,:,:]
        else:
            Thetas = torch.einsum("rikp,rimp->rikm",L_chols,L_chols)
            Thetas_relaxed = Thetas[:,:,None,:,:]+self.relaxations[:,None,None]*torch.eye(Thetas.size(-1),dtype=Thetas.dtype,device=Thetas.device)
            self.L_chols = torch.linalg.cholesky(Thetas_relaxed)
        self.L_inv_chols = torch.linalg.solve_triangular(self.L_chols,torch.eye(self.L_chols.size(-1),dtype=self.L_chols.dtype),upper=False)
        if fp32:
            self.v = self.v.to(torch.float32)
            self.L_chols = self.L_chols.to(torch.float32)
            self.L_inv_chols = self.L_inv_chols.to(torch.float32)
            self.relaxations = self.relaxations.to(torch.float32)
        steps = self.L_chols.size(1) if steps=="all" else steps
        if common_v0:
            i_r_0,i_k_0,i_l_0 = torch.cartesian_prod(
                torch.arange(1),
                torch.arange(1),
                torch.arange(self.relaxations.size(0))).T
            i_r_p,i_k_p,i_l_p = torch.cartesian_prod(
                torch.arange(self.L_chols.size(0)),
                torch.arange(1,steps),
                torch.arange(self.relaxations.size(0))).T
            self.i_r,self.i_k,self.i_l = torch.hstack([i_r_0,i_r_p]),torch.hstack([i_k_0,i_k_p]),torch.hstack([i_l_0,i_l_p])
        else:
            self.i_r,self.i_k,self.i_l = torch.cartesian_prod(
                torch.arange(self.L_chols.size(0)),
                torch.arange(steps),
                torch.arange(self.relaxations.size(0))).T
        self._put_on_device(device)
    def _put_on_device(self, device):
        self.i_r = self.i_r.to(device)
        self.i_k = self.i_k.to(device)
        self.i_l = self.i_l.to(device)
        self.v = self.v.to(device)
        self.L_chols = self.L_chols.to(device)
        self.L_inv_chols = self.L_inv_chols.to(device)
        self.relaxations = self.relaxations.to(device)
    def __getitems__(self, i):
        i = torch.tensor(i,dtype=torch.int)
        ir,ik,il = self.i_r[i],self.i_k[i],self.i_l[i]
        return self.v[ir,ik,:],self.relaxations[il],self.L_chols[ir,ik,il],self.L_inv_chols[ir,ik,il]
    def __len__(self):
        return len(self.i_r)