import torch
import math
from torch import nn
from torch_cluster import radius_graph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.nn import MessagePassing, InstanceNorm

class MPNN_layer(MessagePassing):
    def __init__(self, ninp, nhid):
        super(MPNN_layer, self).__init__()
        self.ninp = ninp
        self.nhid = nhid

        self.message_net_1 = nn.Sequential(nn.Linear(2 * ninp, nhid),
                                           nn.ReLU()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                           nn.ReLU()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(ninp + nhid, nhid),
                                          nn.ReLU()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                          nn.ReLU()
                                          )
        self.norm = InstanceNorm(nhid)

    def forward(self, x, edge_index, batch):
        x = self.propagate(edge_index, x=x)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        message = self.message_net_1(torch.cat((x_i, x_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return update


class MPMC_net(nn.Module):
    def __init__(self, dim, nhid, nlayers, nsamples, nbatch, radius, loss_fn, dim_emphasize, n_projections):
        super(MPMC_net, self).__init__()
        self.enc = nn.Linear(dim,nhid)
        self.convs = nn.ModuleList()
        for i in range(nlayers):
            self.convs.append(MPNN_layer(nhid,nhid))
        self.dec = nn.Linear(nhid,dim)
        self.nlayers = nlayers
        self.mse = torch.nn.MSELoss()
        self.nbatch = nbatch
        self.nsamples = nsamples
        self.dim = dim
        self.n_projections = n_projections
        self.dim_emphasize = torch.tensor(dim_emphasize).long()

        ## random input points for transformation:
        self.x = torch.rand(nsamples * nbatch, dim).to(device)
        batch = torch.arange(nbatch).unsqueeze(-1).to(device)
        batch = batch.repeat(1, nsamples).flatten()
        self.batch = batch
        self.edge_index = radius_graph(self.x, r=radius, loop=True, batch=batch).to(device)

        if loss_fn == 'L2dis':
            self.loss_fn = self.L2discrepancy
        elif loss_fn == 'L2cen':
            self.loss_fn = self.L2center
        elif loss_fn == 'L2ext':
            self.loss_fn = self.L2ext
        elif loss_fn == 'L2per':
            self.loss_fn = self.L2per
        elif loss_fn == 'L2sym':
            self.loss_fn = self.L2sym
        elif loss_fn == 'approx_hickernell':
            if dim_emphasize != None:
                assert torch.max(self.dim_emphasize) <= dim
                self.loss_fn = self.approx_hickernell
        else:
            raise ValueError(f"Loss function DNE: {loss_fn}")

    def approx_hickernell(self, X):
        X = X.view(self.nbatch, self.nsamples, self.dim)
        disc_projections = torch.zeros(self.nbatch).to(device)

        for i in range(self.n_projections):
            ## sample among non-emphasized dimensionality
            mask = torch.ones(self.dim, dtype=bool)
            mask[self.dim_emphasize - 1] = False
            remaining_dims = torch.arange(1, self.dim + 1)[mask]
            projection_dim = remaining_dims[torch.randint(low=0, high=remaining_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(self.dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])
            ## sample among emphasized dimensionality
            remaining_dims = torch.arange(1, self.dim + 1)[self.dim_emphasize - 1]
            projection_dim = remaining_dims[torch.randint(low=0, high=remaining_dims.size(0), size=(1,))].item()
            projection_indices = torch.randperm(self.dim)[:projection_dim]
            disc_projections += self.L2discrepancy(X[:, :, projection_indices])

        return disc_projections

    # def returnx(self):
    #     return self.x
    
    # def bark(self):
    #     print("woof")


    def L2discrepancy(self, x):
        N = x.size(1) 
        dim = x.size(2)
        prod1 = 1. - x ** 2.
        prod1 = torch.prod(prod1, dim=2) #multiplying across second dimenstion of x (dim)
        sum1 = torch.sum(prod1, dim=1) #summing across second dimension of x (number of points in each batch)
        pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
        product = torch.prod(1 - pairwise_max, dim=3)
        sum2 = torch.sum(product, dim=(1, 2))
        one_dive_N = 1. / N
        out = torch.sqrt(
            math.pow(3., -dim) 
            - one_dive_N * math.pow(2., 1. - dim) * sum1 
            + 1. / math.pow(N, 2.) * sum2)
        return out

    def L2center(self, x):
        N = x.size(1)
        dim = x.size(2)

        prod1 = 0.5*(torch.abs(x - 0.5) - (torch.abs(x - 0.5))**2.)
        prod1 = torch.prod(prod1, dim=2)
        sum1 = torch.sum(prod1, dim=1)

        prod2 = 0.5*(torch.abs(x[:, :, None, :] - 0.5) 
                     + torch.abs(x[:, None, :, :] - 0.5) 
                     - torch.abs(x[:, :, None, :] - x[:, None, :, :]))
        product = torch.prod(prod2, dim=3)
        sum2 = torch.sum(product, dim=(1, 2))
        two_div_N = 2. / N
        
        out = math.pow(12., -dim) - two_div_N * sum1 + 1. / math.pow(N, 2.) * sum2
        return out
    
    def L2ext(self, x):
        N = x.size(1)
        dim = x.size(2)

        prod1 = 0.5*(x - x**2.)
        prod1 = torch.prod(prod1, dim = 2)
        sum1 = torch.sum(prod1, dim = 1)

        prod2 = torch.min(x[: ,: ,None ,: ], x[: ,None ,: ,: ]) - x[: ,: ,None ,: ] * x[: ,None ,: ,: ]
        product = torch.prod(prod2, dim = 3)
        sum2 = torch.sum(product, dim = (1,2))

        out = math.pow(12., -dim) - (2. / N) * sum1 + math.pow(N, - 2.) * sum2
        return out
    
    def L2per(self, x):
        N = x.size(1)
        dim = x.size(2)

        prod2 = 0.5 - torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]) + (x[: ,: ,None ,: ] - x[: ,None ,: ,: ])**2
        product = torch.prod(prod2, dim = 3)
        sum2 = torch.sum(product, dim = (1,2))

        out = - math.pow(3., -dim) + math.pow(N, - 2.) * sum2
        return out
    
    def L2sym(self, x): 
        N = x.size(1)
        dim = x.size(2)

        prod1 = 0.5*(x - x**2.)
        prod1 = torch.prod(prod1, dim = 2)
        sum1 = torch.sum(prod1, dim = 1)

        prod2 = 0.25 * (1 - 2 * torch.abs(x[: ,: ,None ,: ] - x[: ,None ,: ,: ]))
        product = torch.prod(prod2, dim = 3)
        sum2 = torch.sum(product, dim = (1,2))

        out = math.pow(12., -dim) - (2. / N) * sum1 + math.pow(N, - 2.) * sum2
        return out
    
    
    def forward(self):
        X = self.x
        edge_index = self.edge_index

        X = self.enc(X)
        for i in range(self.nlayers):
            X = self.convs[i](X,edge_index,self.batch)
        X = torch.sigmoid(self.dec(X))  ## clamping with sigmoid needed so that warnock's formula is well-defined
        X = X.view(self.nbatch, self.nsamples, self.dim)
        loss = torch.mean(self.loss_fn(X))
        return loss, X
