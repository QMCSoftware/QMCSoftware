from types import SimpleNamespace
from qmcpy.discrete_distribution._discrete_distribution import LD
from qmcpy.util import ParameterError
from .utils import L2discrepancy, L2center, L2ext, L2per, L2sym, hickernell_all_emphasized
from .models import *
from tqdm import tqdm 
import torch
import numpy as np
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MPMC(LD): 
    def __init__(self, randomize = None, seed = None, dimension = 2,
                  replications = 1, d_max = None, lr = 0.001, nlayers = 3, weight_decay = 1e-6, nhid = 32,
            epochs = 2000, start_reduce = 1000, radius = 0.35,
            loss_fn = 'L2dis', dim_emphasize = [1], n_projections = 15):
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
        #what to include in this 
        self.parameters = ['dim', 'randomize', 'loss_fn', 'epochs', 'lr', 'nhid']
        self.dim = dimension
        self.lr = lr
        self.nlayers = nlayers
        self.weight_decay = weight_decay
        self.nhid = nhid
        self.epochs = epochs
        self.start_reduce = start_reduce
        self.radius = radius
        self.loss_fn = loss_fn
        self.dim_emphasize = dim_emphasize
        self.n_projections = n_projections
        self.d_max = dimension

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
        
        print(f"\nGenerating {n} samples with MPMC...")

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
                    'dim_emphasize': self.dim_emphasize, # emphasized dimensionalities for projections
                    'n_projections': self.n_projections  # number of projections (for approx_hickernell)
                    }

        #generate points  
        d = self.dim
        r = self.replications
        x = np.empty([r, n, d])
        #if points are already generated: 
        if n in [16, 32, 64, 128, 256]:
            #x = filename smth smth 
            print("x = points already trained")
        else:
            x = self.train(SimpleNamespace(**model_params))


        #randomize
        if self.randomize == "FALSE":
            assert return_unrandomized is False, "cannot return_unrandomized when randomize='FALSE'"
            if r==1: x=x[0]
            return x 
        elif self.randomize == "SHIFT":
            xr = np.empty([r,n,d])
            #randomize smth smth 
            #in lattice, qmctools used for randomizeshift and point generation order 
            xr = (x[:, :, np.newaxis] + self.shift) % 1
            if r==1: xr=xr[0]
            #return both shifted and unshifted if user wants unrandomized, else just randomized
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
    
    #creates new but similar object 
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
                        radius = args.radius, loss_fn = args.loss_fn, dim_emphasize = args.dim_emphasize, n_projections = args.n_projections).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = 10000.
        patience = 0
        #set of points to return later 
        end_result = None

        ## could be tuned for better performance
        start_reduce = 100000
        reduce_point = 10

        for epoch in tqdm(range(args.epochs), desc = f"Training: N={args.nsamples}, nhid={args.nhid}, loss={args.loss_fn}"):
            if (epoch % 5000 == 0):
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

if __name__ == '__main__':
    print("\n Running MPMC example \n")
    mpmc_gen = MPMC(dimension=2, loss_fn='L2dis', epochs=500, nhid=6, randomize = 'shift', seed = 8)
    points = mpmc_gen.gen_samples(n = 50)

    print("\n--- Generation Complete ---")
    print(f"Shape of generated points: {points.shape}")
    print("First 5 points:\n", points[:5])
    print(mpmc_gen)
