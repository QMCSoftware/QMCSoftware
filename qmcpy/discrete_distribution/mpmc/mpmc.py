from types import SimpleNamespace
from .._discrete_distribution import LD
from ...util import ParameterError, ParameterWarning
from utils import L2discrepancy, hickernell_all_emphasized, L2center, L2ext, L2per, L2sym
from models import *
from tqdm import tqdm 
import torch
import numpy as np
from torch_cluster import radius_graph
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MPMC(LD): 
    def __init__(self, dimension = 2, randomize = None, seed = None, 
                  replications = 1, **kwargs):
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
        self.mimics = 'StdUniform'
        self.low_discrepancy = True
        self.hyper_params = {
            'lr': 0.001, 'nlayers': 3, 'weight_decay': 1e-6, 'nhid': 32,
            'epochs': 2000, 'start_reduce': 1000, 'radius': 0.35, 
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
        
        #randomization
        self.randomize = str(randomize).upper()
        if self.randomize == "TRUE" : self.randomize = "SHIFT"
        if self.randomize == "NONE" | "NO": self.randomize = "FALSE"
        assert self.randomize in ["SHIFT", "FALSE"]
        if self.randomize == "SHIFT":
            #matrix of size #replications * dimension (one shift for each rep/dim)
            self.shift = self.rng.uniform(size = (replications, self.d))
        self.replications = replications


    def gen_samples(self, n = None, warn = True, return_unrandomized = False, loss_fn = 'L2dis'):
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
        
        print(f"\nGenerating {n} samples with MPMC...")

        model_params = {
            'dim': self.d,
            'nsamples': n,
            **self.hyper_params
        }

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
            x = self.train(**model_params)


        #randomize
        if self.randomize == "FALSE":
            assert return_unrandomized is False, "cannot return_unrandomized when randomize='FALSE'"
            return x 
        elif self.randomize == "SHIFT":
            xr = np.empty(r,n,d)
            #randomize smth smth 
            if r==1: xr=xr[0]
            #in lattice, qmctools used for randomizeshift and point generation order 
            xr = (x[:, :, np.newaxis] + self.shift) % 1
            #return both shifted and unshifted if user wants unrandomized, else just randomized
            return (xr, x) if return_unrandomized else xr 
        else: 
            raise ParameterError("incorrect randomize parsing in lattice gen_samples")
            
    
    def _get_params_interactively(self):
        """Prompt user for parameters via the command line."""
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
        print("-" * 50)


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

        for epoch in tqdm(range(args.epochs), desc = f"Training: N={args.nsamples}, nhid={args.nhid}, loss={args.loss_fn}"):
            if (epoch % 10000 == 0):
                print(f"epoch: {epoch}")

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

                    ## save MPMC points:
                    y = y.detach().cpu().numpy()
                    #y (from X) has batch x nsamples x dim dimension 
                    endresult = y

                if (min_discrepancy > best_loss and (epoch + 1) >= args.start_reduce):
                    patience += 1

                if (epoch + 1) >= args.start_reduce and patience == reduce_point:
                    patience = 0
                    args.lr /= 10.
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr

                if (args.lr < 1e-6):
                    break
            
        return endresult
    
    


#temporary for testing: 
m = MPMC()
print(m.gen_samples(n = 10))