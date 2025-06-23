# this code implemments mpmc as an object like in the lattice
# note only working with pytorch and higher version of python for me
# - A


# need to change to numpy?

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import types
import math
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing, InstanceNorm
from .._discrete_distribution import LD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Discrepancy functions to call
def L2discrepancy(x):
    if x.dim() == 2: x = x.unsqueeze(0)
    N, dim = x.size(1), x.size(2)
    if N == 0: return torch.tensor(0.0, device=x.device)
    term1 = torch.tensor(math.pow(3., -dim), device=x.device)
    prod1 = torch.prod(1. - x ** 2., dim=2)
    sum1 = torch.sum(prod1, dim=1)
    term2 = (2. / N) * math.pow(2., 1-dim) * sum1
    pairwise_max = torch.maximum(x[:, :, None, :], x[:, None, :, :])
    product = torch.prod(1. - pairwise_max, dim=3)
    sum2 = torch.sum(product, dim=(1, 2))
    term3 = (1. / math.pow(N, 2.)) * sum2
    epsilon = 1e-8
    out = torch.sqrt(torch.clamp(term1 - term2 + term3, min=epsilon))
    return out

def L2center(x):
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

def L2ext(x):
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

def L2per(x):
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




def hickernell_all_emphasized(x, n_projections, dim_emphasize):
    nbatch, nsamples, dim = x.shape
    disc_projections = torch.zeros(nbatch, device=x.device)
    dim_emphasize_tensor = torch.tensor(dim_emphasize, device=x.device, dtype=torch.long) - 1

    for _ in range(n_projections):
        mask = torch.ones(dim, dtype=bool, device=x.device)
        if dim_emphasize_tensor.numel() > 0:
            mask[dim_emphasize_tensor] = False
        
        remaining_dims = torch.arange(0, dim, device=x.device)[mask]
        if len(remaining_dims) > 0:
            projection_dim = torch.randint(low=1, high=len(remaining_dims) + 1, size=(1,)).item()
            perm = torch.randperm(len(remaining_dims), device=x.device)
            projection_indices = remaining_dims[perm[:projection_dim]]
            disc_projections += L2discrepancy(x[:, :, projection_indices])
        
        if len(dim_emphasize_tensor) > 0:
            projection_dim = torch.randint(low=1, high=len(dim_emphasize_tensor) + 1, size=(1,)).item()
            perm = torch.randperm(len(dim_emphasize_tensor), device=x.device)
            projection_indices = dim_emphasize_tensor[perm[:projection_dim]]
            disc_projections += L2discrepancy(x[:, :, projection_indices])
            
    return disc_projections



# Model definitions
class MPNN_layer(MessagePassing):
    def __init__(self, ninp, nhid):
        super(MPNN_layer, self).__init__(aggr='add')
        self.message_net_1 = nn.Sequential(nn.Linear(2 * ninp, nhid), nn.ReLU())
        self.message_net_2 = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU())
        self.update_net_1 = nn.Sequential(nn.Linear(ninp + nhid, nhid), nn.ReLU())
        self.update_net_2 = nn.Sequential(nn.Linear(nhid, nhid), nn.ReLU())
        self.norm = InstanceNorm(nhid)

    def forward(self, x, edge_index):
        updated_x = self.propagate(edge_index, x=x)
        norm_x = self.norm(updated_x)
        return norm_x + x 

    def message(self, x_i, x_j):
        return self.message_net_2(self.message_net_1(torch.cat((x_i, x_j), dim=-1)))

    def update(self, aggregated_message, x):
        return self.update_net_2(self.update_net_1(torch.cat((x, aggregated_message), dim=-1)))

class MPMC_net(nn.Module):
    def __init__(self, dim, nhid, nlayers, nsamples, radius, loss_fn, dim_emphasize, n_projections, nbatch=1):
        super(MPMC_net, self).__init__()
        self.nbatch, self.nsamples, self.dim, self.radius = nbatch, nsamples, dim, radius
        self.loss_fn_name, self.n_projections, self.dim_emphasize = loss_fn, n_projections, dim_emphasize
        
        self.enc = nn.Linear(dim, nhid)
        self.convs = nn.ModuleList([MPNN_layer(nhid, nhid) for _ in range(nlayers)])
        self.dec = nn.Linear(nhid, dim)
        self.initial_points = nn.Parameter(torch.rand(nsamples * nbatch, dim, device=device))

    def forward(self):
        x = self.initial_points
        batch = torch.arange(self.nbatch, device=device).unsqueeze(-1).repeat(1, self.nsamples).flatten()
        edge_index = radius_graph(x, r=self.radius, batch=batch, loop=True)
        x_encoded = self.enc(x)
        for conv in self.convs: x_encoded = conv(x_encoded, edge_index)
        x_decoded = torch.sigmoid(self.dec(x_encoded))
        X = x_decoded.view(self.nbatch, self.nsamples, self.dim)
        
        if self.loss_fn_name == 'approx_hickernell':
            loss = torch.mean(hickernell_all_emphasized(X, self.n_projections, self.dim_emphasize))
        else:
            loss = torch.mean(globals()[self.loss_fn_name](X))
        return loss, X


# QMC Implemmentation
class MPMC(LD):
    """
    Generates point sets by optimizing for low discrepancy using a
    Message Passing Neural Network.
    
    >>> mpmc = MPMC(dimension=2, loss_fn='L2dis', epochs=100)
    >>> points = mpmc.gen_samples(n=50)
    >>> points.shape
    (50, 2)
    >>> mpmc
    MPMC (LD Object)
        dimension       2
        randomize       SHIFT
        loss_fn         L2dis
        epochs          100
        lr              0.001
        nhid            32
    """


    def __init__(self, dimension=None, randomize='shift', seed=None, **kwargs):
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        

        self.hyper_params = {
            'lr': 0.001, 'nlayers': 3, 'weight_decay': 1e-6, 'nhid': 32,
            'epochs': 2000, 'start_reduce': 1000, 'radius': 0.35, 'nbatch': 1,
            'loss_fn': 'L2dis', 'dim_emphasize': [1], 'n_projections': 15
        }
        
        # If no dimension passed, use interactive
        if dimension is None:
            self._get_params_interactively()
            dimension = self.hyper_params['dimension']
        
        self.hyper_params.update(kwargs)
        self.hyper_params['dimension'] = dimension 

        for key, val in self.hyper_params.items():
            setattr(self, key, val)

        self.parameters = ['dimension', 'randomize', 'loss_fn', 'epochs', 'lr', 'nhid']
        super(MPMC, self).__init__(dimension, seed)

        # RANDOMIZATION HERE
        self.randomize = str(randomize).upper()
        if self.randomize not in ["SHIFT", "FALSE"]: self.randomize = "SHIFT"


    # need to add all parameters?
    def _get_params_interactively(self):
        print("-" * 50)
        print("MPMC Interactive Parameter Setup")
        print("Press Enter to use the default value shown in [].")
        print("-" * 50)


        param_map = {
            'dimension': ('Dimension of the point set', int),
            'loss_fn': ('Loss function (e.g., L2dis)', str),
            'epochs': ('Number of training epochs', int),
            'lr': ('Learning rate', float),
            'nhid': ('Number of hidden features', int),
        }
        
        for key, (prompt, type_cast) in param_map.items():
            default = self.hyper_params.get(key, 'N/A')
            while True:
                try:
                    user_input = input(f"- {prompt} [{default}]: ")
                    if user_input == "":
                        break
                    else:
                        self.hyper_params[key] = type_cast(user_input)
                        break
                except ValueError:
                    print(f"  Invalid input. Please enter a value of type '{type_cast.__name__}'.")


    def gen_samples(self, n=None, n_min=0, n_max=None, replications=1):
        if n: n_max = n
        if not n_max: raise ValueError("n or n_max must be provided.")

        print(f"\nGenerating {n_max} samples with MPMC...")
        
        model_params = {
            'dim': self.d, 
            'nsamples': n_max,
            **self.hyper_params 
        }

        model = MPMC_net(**model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        progress_bar = tqdm(range(self.epochs), desc=f"Training (N={n_max}, loss={self.loss_fn})")
        for epoch in progress_bar:
            model.train()
            optimizer.zero_grad()
            training_loss, _ = model()
            training_loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{training_loss.item():.6f}")
        
        model.eval()
        with torch.no_grad():
            _, final_points = model()
        
        points = final_points.cpu().numpy() 
        
        # RANDOMIZATION HERE
        if self.randomize == "SHIFT":
            shifts = self.rng.uniform(size=(replications, self.d))
            all_points = np.zeros((replications, n_max, self.d))
            for i in range(replications):
                all_points[i] = (points[0] + shifts[i]) % 1.0
            return np.squeeze(all_points) 
        else:
            return np.squeeze(points) 

    def _spawn(self, child_seed, dimension):
        return MPMC(
            dimension=dimension,
            randomize=self.randomize,
            seed=child_seed,
            **self.hyper_params)


# test run
if __name__ == '__main__':

    print("Running MPMC QMCPy Test")

    
    mpmc_gen = MPMC(dimension=2, loss_fn='L2dis', epochs=500, nhid=64)

    points = mpmc_gen.gen_samples(n=64)
    
    print("\n--- Generation Complete ---")
    print(f"Shape of generated points: {points.shape}")
    print("First 5 points:\n", points[:5])
    
    print("\nGenerator Configuration:")
    print(mpmc_gen)

    print("\n" + "="*80)
    print("Example with 3 replications")
    print("="*80)
    mpmc_gen_rep = MPMC(dimension=3, epochs=200, seed=1234)
    points_rep = mpmc_gen_rep.gen_samples(n=32, replications=3)
    print("\n--- Generation Complete ---")
    print(f"Shape of generated points: {points_rep.shape}")
