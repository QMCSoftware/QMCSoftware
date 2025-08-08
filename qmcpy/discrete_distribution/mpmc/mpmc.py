from types import SimpleNamespace
from qmcpy.discrete_distribution._discrete_distribution import LD
from qmcpy.util import ParameterError
from .utils import L2star, L2ctr, L2ext, L2per, L2sym, L2asd, L2mix, L2star_weighted, L2ctr_weighted, L2ext_weighted, L2per_weighted, L2sym_weighted, L2asd_weighted, L2mix_weighted
# from utils import L2star, L2ctr, L2ext, L2per, L2sym, L2asd, L2mix, L2star_weighted, L2ctr_weighted, L2ext_weighted, L2per_weighted, L2sym_weighted, L2asd_weighted, L2mix_weighted
# from models import *
from .models import *
from tqdm import tqdm 
import torch
import numpy as np
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
from os.path import dirname, abspath, isfile
from pathlib import Path


#temporarily use nbatch instead of replication bc of inherited error
class MPMC(LD): 
    def __init__(self, randomize = 'shift', seed = None, dimension = 2,
                  replications = 1, d_max = None, lr = 0.001, nlayers = 3, weight_decay = 1e-6, nhid = 32,
                  epochs = 10000, start_reduce = 8000, radius = 0.35, nbatch = 1,
                  loss_fn = 'L2dis', weights = None):
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

            weights (list or array??): array of weights for weighted discrepancies
            
            replications (int): number of IID randomizations of a pointset
        """
        #necessary 
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        #what to include in this 
        self.parameters = ['dim', 'randomize', 'loss_fn', 'epochs', 'lr', 'nhid']
        self.dim = dimension
        self.nbatch = nbatch
        self.lr = lr
        self.nlayers = nlayers
        self.weight_decay = weight_decay
        self.nhid = nhid
        self.epochs = epochs
        self.start_reduce = start_reduce
        self.radius = radius
        self.loss_fn = loss_fn
        self.weights = weights
        if weights != None:
            self.weights = torch.tensor(weights, device=device, dtype = torch.float32).long() 
        self.d_max = dimension

        if ((self.loss_fn[-8:] == 'weighted') & (self.weights == None)):
            raise ValueError(f"Must specify weights for weighted loss function")

        if ((self.weights == None) & (self.d_max > 5)):
            warnings.warn("Weights are recommended for dimension 5 and above")

        if ((self.weights != None ) & (self.loss_fn[-8:] != 'weighted')):
            print("Since you included weights, you will be using the weighted version of this discrepancy. Set weights = None if you want the unweighted discrepancy.")
            self.loss_fn = self.loss_fn + "_weighted"

        super(MPMC, self).__init__(dimension, seed)
        
        #randomization
        self.randomize = str(randomize).upper()
        if self.randomize == "TRUE" : self.randomize = "SHIFT"
        if (self.randomize == "NONE") | (self.randomize == "NO"): self.randomize = "FALSE"
        assert self.randomize in ["SHIFT", "FALSE"]
        if self.randomize == "SHIFT":
            #matrix of size #replications * dimension (one shift for each rep/dim)
            self.shift = self.rng.uniform(size = (replications, self.d))
        self.replications = replications


    def gen_samples(self, n = None, warn = True, return_unrandomized = False):
        """
        IMPLEMENT ABSTRACT METHOD to generate samples from this discrete distribution.

        Args:
            n (int): if n is supplied
            return_unrandomized (bool): return samples without randomization as 2nd return value.
                Will not be returned if randomize=False.
        Returns:
            ndarray: replicatsions x n x d array of samples
        """
        #error if n is None: 
        if (n == None ):
            raise ValueError("Must provide n number of points to generate")
        
        # print(f"\nGenerating {n} samples with MPMC...")

        model_params = {
                    'lr': self.lr,                      # learning rate
                    'nlayers': self.nlayers,           # number of GNN layers
                    'weight_decay': self.weight_decay, # weight decay (L2 regularization)
                    'nhid': self.nhid,                 # number of hidden features in the GNN
                    'nbatch': self.replications,             # number of point sets in a batch
                    'epochs': self.epochs,             # number of training epochs
                    'start_reduce': self.start_reduce, # epoch to start reducing learning rate
                    'radius': self.radius,             # radius for GNN neighborhood
                    'nsamples': n,         # number of samples in each point set
                    'dim': self.dim,                   # dimensionality of the points
                    'loss_fn': self.loss_fn,           # loss function to use
                    'weights': self.weights, # emphasized dimensionalities for projections
                    }

        #generate points  
        d = self.dim
        r = self.replications
        x = np.empty([r, n, d])

        #in case pathlib not allowed, this uses everything lattice uses. 
        repos = np.lib.npyio.DataSource()
        head = "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/pregenerated_pointsets/mpmc/"
        #change variables to be more compatible
        # wd = str(self.weight_decay).replace('e-0', 'e-').replace('e+0', 'e+')
        # if self.weights != None:
        #     weights = ''.join(map(str, self.weights.tolist()))
        # else:
        #     weights = None
        filename = (
            f"dim_{self.dim}.nsamples_{n}.nbatch_{self.nbatch}"
            f".Lf{self.loss_fn}.b_1.txt"
        )
        link = f"{head}{filename}"
        if (repos.exists(link)):
            #ask if user wants pretrained points
            # proceed = input("Points already trained with same dimensions, samples, batches, and loss function. *warning: other parameters may not be the same*. Proceed with pretrained points?(y/n) ")
            # proceed = str(proceed).upper()
            # if proceed == 'yes': proceed = 'Y'
            # if proceed == 'no': proceed = 'N'
            # assert proceed in ['Y', 'N']
            # if (proceed == 'Y'):
            #     temp = []
            #     for b in range(1, self.nbatch + 1): 
            #         filename = (
            #             f"dim_{self.dim}.nsamples_{n}.nbatch_{self.nbatch}"
            #             f".Lf{self.loss_fn}.b_{b}.txt"
            #         )
            #         link = f"{head}{filename}"
            #         #extract from link 
            #         datafile = repos.open(link)
            #         data = np.loadtxt(datafile, skiprows = 15)
            #             #strip comments, turn into numpy array, extract d_limit, n_limit
            #         #add to temp (many batches)
            #         temp.append(data)
            #     x = np.array(temp)
            # else: 
            #     x = self.train(SimpleNamespace(**model_params))

            temp = []
            for b in range(1, self.nbatch + 1): 
                filename = (
                    f"dim_{self.dim}.nsamples_{n}.nbatch_{self.nbatch}"
                    f".Lf{self.loss_fn}.b_{b}.txt"
                )
                link = f"{head}{filename}"
                #extract from link 
                datafile = repos.open(link)
                data = np.loadtxt(datafile, skiprows = 15)
                    #strip comments, turn into numpy array, extract d_limit, n_limit
                #add to temp (many batches)
                temp.append(data)
            x = np.array(temp)
        else:
            x = self.train(SimpleNamespace(**model_params))
        #randomize
        if self.randomize == "FALSE":
            assert return_unrandomized is False, "cannot return_unrandomized when randomize='FALSE'"
            if self.nbatch==1: x=x[0]
            return x 
        elif self.randomize == "SHIFT":
            xr = np.empty([r,n,d])
            xr = (x + self.shift) % 1
            if self.nbatch==1: xr=xr[0]
            return (xr, x) if return_unrandomized else xr 
        else: 
            raise ParameterError("incorrect randomize parsing in lattice gen_samples")
            

    def pdf(self, x):
        """ pdf of a standard uniform """
        return np.ones(x.shape[:-1], dtype=float)
    
    def __repr__(self):
        out = f"{self.__class__.__name__} Generator Object\n"
        for p in self.parameters:
            p_val = getattr(self,p)
            out += f"    {p:<15} {str(p_val)}\n"
        return out
    

    def _spawn(self, child_seed, dimension): 
        """ 
        assign parameters 
        """
        return MPMC(
            dimension=dimension,
            randomize=self.randomize,
            dim_emph= self.dim_emph,
            seed=child_seed,
            replications=self.replications)

    #returns an NP array of points (trained)
    def train(self, args):
        model = MPMC_net(dim = args.dim, nhid = args.nhid, nlayers = args.nlayers, nsamples = args.nsamples, nbatch = args.nbatch,
                        radius = args.radius, loss_fn = args.loss_fn, weights = args.weights).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = 10000.
        patience = 0
        end_result = None

        ## could be tuned for better performance
        start_reduce = 100000
        reduce_point = 10

        for epoch in tqdm(range(args.epochs), desc = f"Training: N={args.nsamples}, nhid={args.nhid}, loss={args.loss_fn}"):

            model.train()
            optimizer.zero_grad()
            loss, X = model()
            loss.backward()
            # print(f"YOYOYO {X.clone()[:5]}")
            optimizer.step()

            if epoch % 100 ==0:
                y = X.clone()

                if args.loss_fn[-8:] == 'weighted':
                    batched_discrepancies = globals()[args.loss_fn](y.detach(), args.weights)
                else: 
                    batched_discrepancies = globals()[args.loss_fn](y.detach())
                min_discrepancy, mean_discrepancy = torch.min(batched_discrepancies).item(), torch.mean(batched_discrepancies).item()

                if min_discrepancy < best_loss:
                    best_loss = min_discrepancy
                    y = y.detach().cpu().numpy()
                    end_result = y

                if (min_discrepancy > best_loss and (epoch + 1) >= args.start_reduce):
                    patience += 1

                if (epoch + 1) >= args.start_reduce and patience == reduce_point:
                    patience = 0
                    args.lr /= 10.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr

                if (args.lr < 1e-6):
                    break
            
        return end_result
    
    


#temporary for testing: 

if __name__ == '__main__':
    print("\n Running MPMC example \n")
    # wpoints = MPMC(dimension = 8, epochs = 500, weights = [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128], loss_fn = 'L2sym')

    w = [0]*52
    w[0:3] = [1,1,1]
    wpoints = MPMC(dimension = 52, nbatch=1, weights = w, loss_fn = 'L2asd')
    # wpoints = MPMC(dimension = 2, loss_fn = 'L2asd', epochs = 200000, start_reduce = 100000)
 
    w = wpoints.gen_samples (n = 32)
    print("\n--- Generation Complete ---")
    print(f"Shape of generated points: {w.shape}")
    print("First 5 points:\n", w[:5])
    print(wpoints)
