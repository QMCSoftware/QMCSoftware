from .._discrete_distribution import LD
import torch
import math
from torch import nn
from torch_cluster import radius_graph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch_geometric.nn import MessagePassing, InstanceNorm
from numpy import *

class MPMC(LD): 
    def __init__(self, dimension = 2, randomize = 'shift', seed = None, 
                 dim_emph = None, d_max = None, replications = 1):
        """
        Args:
            dimension (int or ndarray): dimension of the generator.
                If an int is passed in, use sequence dimensions [0,...,dimensions-1].
                If a ndarray is passed in, use these dimension indices in the sequence.
                Note that this is not relevant for IID generators.

            seed (int or numpy.random.SeedSequence): seed to create random number generator

            randomize (bool): If True, apply shift to generated samples.
                Note: Non-randomized lattice sequence includes the origin.

            

        """
        #necessary 
        self.parameters = ['randomize']
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        self.replications = replications
        self.d_max = d_max

        super(MPMC, self).__init__(dimension, seed)
        
        #randomization
        
        



    def gen_samples(self, n = None, warn = True, return_unrandomized = False,):
        """
        IMPLEMENT ABSTRACT METHOD to generate samples from this discrete distribution.

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples.
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            return_unrandomized (bool): return samples without randomization as 2nd return value.
                Will not be returned if randomize=False.
        Returns:
            ndarray: replicatsions x n x d array of samples
        """
        #generate points  
        d = self.d
        r = self.replications
        x = empty(r, n, d)
        #randomize
        if self.randomize == "FALSE":
            return x 
        elif self.randomize == "SHIFT":
            xr = empty(r,n,d)
            return (xr, x)
    
        
    
    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[:-1], dtype=float)
        
    
    def _spawn(self, dimension): 
        """ 
        assign parameters 
        """
    
    
    



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


        #we decide
        self.nlayers = nlayers
        self.mse = torch.nn.MSELoss()
        #replication
        self.nbatch = nbatch
        #n- gen sample
        self.nsamples = nsamples
        #dimension- 
        self.dim = dim
        self.n_projections = n_projections
        #user input in gen samples 
        self.dim_emphasize = torch.tensor(dim_emphasize).long()

        ## random input points for transformation:
        self.x = torch.rand(nsamples * nbatch, dim).to(device)
        batch = torch.arange(nbatch).unsqueeze(-1).to(device)
        batch = batch.repeat(1, nsamples).flatten()
        self.batch = batch
        self.edge_index = radius_graph(self.x, r=radius, loop=True, batch=batch).to(device)

        #which one do we use? 
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

    def train(args):
        print("intrain")
        model = MPMC_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                        args.radius, args.loss_fn, args.dim_emphasize, args.n_projections).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = 10000.
        patience = 0

        ## could be tuned for better performance
        start_reduce = 100000
        reduce_point = 10

        # Path('results/dim_' + str(args.dim) + '/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid)).mkdir(parents=True, exist_ok=True)
        # Path('outputs/dim_' + str(args.dim) + '/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid)).mkdir(parents=True, exist_ok=True)

        for epoch in tqdm(range(args.epochs), desc = f"Training: N={args.nsamples}, nhid={args.nhid}, loss={args.loss_fn}"):
            model.train()
            optimizer.zero_grad()
            loss, X = model()
            loss.backward()
            optimizer.step()

            if epoch % 100 ==0:
                y = X.clone()
                if args.loss_fn == 'L2dis':
                    batched_discrepancies = L2discrepancy(y.detach())
                elif args.loss_fn == 'L2cen':
                    batched_discrepancies = L2center(y.detach())
                elif args.loss_fn == 'L2ext':
                    batched_discrepancies = L2ext(y.detach())
                elif args.loss_fn == 'L2per':
                    batched_discrepancies = L2per(y.detach())
                elif args.loss_fn == 'L2sym':
                    batched_discrepancies = L2sym(y.detach())
                elif args.loss_fn == 'approx_hickernell':
                    ## compute sum over all projections with emphasized dimensionality:
                    batched_discrepancies = hickernell_all_emphasized(y.detach(),args.dim_emphasize)
                else:
                    print('Loss function not implemented')
                min_discrepancy, mean_discrepancy = torch.min(batched_discrepancies).item(), torch.mean(batched_discrepancies).item()

                if min_discrepancy < best_loss:
                    best_loss = min_discrepancy
                    f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                    f.write(str(best_loss) + '\n')
                    f.close()

                    ## save MPMC points:
                    PATH = 'outputs/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+ '/nhid_' +str(args.nhid)+ '/Lf'+str(args.loss_fn) + '.npy'
                    y = y.detach().cpu().numpy()
                    np.save(PATH,y)

                if (min_discrepancy > best_loss and (epoch + 1) >= args.start_reduce):
                    patience += 1

                if (epoch + 1) >= args.start_reduce and patience == reduce_point:
                    patience = 0
                    args.lr /= 10.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr

                if (args.lr < 1e-6):
                    f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                    f.write('### epochs: '+str(epoch) + '\n')
                    f.close()
                    break