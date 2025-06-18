from types import SimpleNamespace
from .._discrete_distribution import LD
import torch
import numpy as np
from torch import nn
from torch_cluster import radius_graph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
from utils import L2discrepancy, hickernell_all_emphasized, L2center, L2ext, L2per, L2sym
from models import *
from tqdm import tqdm 


class MPMC(LD): 
    def __init__(self, dimension = 2, randomize = None, seed = None, 
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

            d_max (int): max dimension

            dim_emph (list or array??): emphasized dimensions for approx_hickernell 
            
            replications (int): number of IID randomizations of a pointset
        """
        #necessary 
        #what is this list of parameters for? 
        self.parameters = ['randomize']
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        self.replications = replications
        self.d_max = d_max

        super(MPMC, self).__init__(dimension, seed)

        
        
        #randomization


    def gen_samples(self, n = None, return_unrandomized = False):
        """
        IMPLEMENT ABSTRACT METHOD to generate samples from this discrete distribution.

        Args:
            n (int): if n is supplied
            return_unrandomized (bool): return samples without randomization as 2nd return value.
                Will not be returned if randomize=False.
        Returns:
            ndarray: replicatsions x n x d array of samples
        """
        #generate points  
        d = self.d
        r = self.replications
        x = np.empty(r, n, d)
        #if points are already generated: 
        if n in [16, 32, 64, 128, 256]:
            #x = filename smth smth 
            print("x = points already trained")
        #else train data using model: 
        else:
            args = {
                'lr': 0.001,                  # learning rate
                'nlayers': 3,                 # number of GNN layers
                'weight_decay': 1e-6,         # weight decay (L2 regularization)
                'nhid': 32,                  # number of hidden features in the GNN
                'nbatch': 1,                  # number of point sets in a batch
                'epochs': 200000,             # number of training epochs
                'start_reduce': 100000,       # epoch to start reducing learning rate
                'radius': 0.35,               # radius for GNN neighborhood
                'nsamples': n,               # number of samples in each point set
                'dim': 2,                     # dimensionality of the points
                'loss_fn': 'L2dis',           # loss function to use
                'dim_emphasize': [1],         # emphasized dimensionalities for projections
                'n_projections': 15           # number of projections (for approx_hickernell)
            }

            args = SimpleNamespace(**args)
            x = self.train(args)


        #randomize
        if self.randomize == "FALSE":
            return x 
        elif self.randomize == "SHIFT":
            xr = np.empty(r,n,d)
            #randomize smth smth 
            return (xr, x)
    
        
    
    def pdf(self, x):
        """ pdf of a standard uniform """
        return np.ones(x.shape[:-1], dtype=float)
        
    
    def _spawn(self, child_seed, dimension): 
        """ 
        assign parameters 
        """
        return MPMC(
            dimension=dimension,
            randomize=self.randomize,
            dim_emph= self.dim_emph,
            seed=child_seed,
            d_max=self.d_max,
            replications=self.replications)

    #returns an NP array of points (trained)
    def train(self, args):
        model = MPMC_net(args.dim, args.nhid, args.nlayers, args.nsamples, args.nbatch,
                        args.radius, args.loss_fn, args.dim_emphasize, args.n_projections).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = 10000.
        patience = 0
        #set of points to return later 
        end_result = None

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

                #
                if min_discrepancy < best_loss:
                    best_loss = min_discrepancy

                    #REPLACE POINT SAVE IN FILE TO POINT SAVE TO X 
                    # f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                    # f.write(str(best_loss) + '\n')
                    # f.close()

                    ## save MPMC points:
                    y = y.detach().cpu().numpy()
                    endresult = y

                if (min_discrepancy > best_loss and (epoch + 1) >= args.start_reduce):
                    patience += 1

                if (epoch + 1) >= args.start_reduce and patience == reduce_point:
                    patience = 0
                    args.lr /= 10.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr

                if (args.lr < 1e-6):
                    # f = open('results/dim_'+str(args.dim)+'/nsamples_'+str(args.nsamples)+'/nhid_'+str(args.nhid) + '/Lf'+str(args.loss_fn) + '.txt', 'a')
                    # f.write('### epochs: '+str(epoch) + '\n')
                    # f.close()

                    break
            
        return endresult
    
    
    

    