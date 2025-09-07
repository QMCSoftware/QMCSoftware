from types import SimpleNamespace
from qmcpy.discrete_distribution._discrete_distribution import LD
from qmcpy.util import ParameterError
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import warnings

from .utils import (
    L2star, L2ctr, L2ext, L2per, L2sym, L2asd, L2mix,
    L2star_weighted, L2ctr_weighted, L2ext_weighted, L2per_weighted,
    L2sym_weighted, L2asd_weighted, L2mix_weighted,
)
from .models import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_DISCREPANCY = {
    'L2star': L2star, 'L2ctr': L2ctr, 'L2ext': L2ext, 'L2per': L2per, 'L2sym': L2sym, 'L2asd': L2asd, 'L2mix': L2mix,
    'L2star_weighted': L2star_weighted, 'L2ctr_weighted': L2ctr_weighted, 'L2ext_weighted': L2ext_weighted,
    'L2per_weighted': L2per_weighted, 'L2sym_weighted': L2sym_weighted, 'L2asd_weighted': L2asd_weighted,
    'L2mix_weighted': L2mix_weighted,
}

class MPMC(LD):
    """
    Low-discrepancy generator trained by MPMC. Produces nbatch independent pointsets of size n in [0,1]^d.
    """

    def __init__(
        self,
        randomize='shift',
        seed=None,
        dimension=2,
        replications=1,       
        d_max=None,            
        lr=1e-3,
        nlayers=3,
        weight_decay=1e-6,
        nhid=32,
        epochs=50_000,
        start_reduce=40_000,
        radius=0.35,
        nbatch=1,
        loss_fn='L2star',
        weights=None,
    ):
        self.mimics = 'StdUniform'
        self.low_discrepancy = True

        self.parameters = [
            'dim', 'randomize', 'loss_fn', 'epochs', 'lr', 'nlayers', 'nhid',
            'weight_decay', 'radius', 'nbatch'
        ]

        # core config
        self.dim = int(dimension)
        self.lr = float(lr)
        self.nlayers = int(nlayers)
        self.weight_decay = float(weight_decay)
        self.nhid = int(nhid)
        self.epochs = int(epochs)
        self.start_reduce = int(start_reduce)
        self.radius = float(radius)
        self.loss_fn = str(loss_fn)
        self.nbatch = int(nbatch) if nbatch is not None else int(replications)
        self.d_max = self.dim  # kept for compat

        self.weights = None
        if weights is not None:
            # accept list/np/torch → torch.float32 on device
            self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
            if self.weights.dim() != 1 or self.weights.numel() != self.dim:
                raise ValueError(f"weights must be 1-D length d={self.dim}; got {tuple(self.weights.shape)}")

        # ensure weighted function name & presence of weights are consistent
        is_weighted_name = self.loss_fn.endswith('weighted')
        if is_weighted_name and self.weights is None:
            raise ValueError("Must specify `weights` for weighted loss function.")
        if (self.weights is not None) and (not is_weighted_name):
            warnings.warn("`weights` provided; switching to weighted discrepancy.", stacklevel=1)
            self.loss_fn = self.loss_fn + '_weighted'

        if (self.weights is None) and (self.dim > 5):
            warnings.warn("Product coordinate weights are recommended for dimension > 5.", stacklevel=1)

        # init LD base class (sets self.rng, self.d, etc.)
        super(MPMC, self).__init__(dimension=self.dim, seed=seed)

        # randomization mode
        rnd = str(randomize).strip().upper()
        if rnd in ('TRUE', 'SHIFT'):
            self.randomize = 'SHIFT'
        elif rnd in ('FALSE', 'NONE', 'NO'):
            self.randomize = 'FALSE'
        else:
            raise ParameterError(f"randomize must be in {{'shift','false'}}; got '{randomize}'")

        # pre-draw shifts if needed: shape (nbatch, d)
        if self.randomize == 'SHIFT':
            self.shift = self.rng.uniform(size=(self.nbatch, self.d))

        # backward-compat mirror
        self.replications = self.nbatch

    # --------------------------
    # Core API
    # --------------------------
    def gen_samples(self, n=None, warn=True, return_unrandomized=False):
        """
        Generate samples.

        Args:
            n (int): number of points per set.
            return_unrandomized (bool): if SHIFT randomization, also return the base points.
        Returns:
            If nbatch == 1:
                - randomize='FALSE': ndarray (n, d)
                - randomize='SHIFT' & return_unrandomized=False: ndarray (n, d)
                - randomize='SHIFT' & return_unrandomized=True: (ndarray (n,d), ndarray (n,d))
            If nbatch > 1:
                - shapes include the leading (nbatch, ...).
        """
        if n is None:
            raise ValueError("Must provide n number of points to generate.")
        n = int(n)

        # training config
        args = SimpleNamespace(
            lr=self.lr,
            nlayers=self.nlayers,
            weight_decay=self.weight_decay,
            nhid=self.nhid,
            nbatch=self.nbatch,
            epochs=self.epochs,
            start_reduce=self.start_reduce,
            radius=self.radius,
            nsamples=n,
            dim=self.dim,
            loss_fn=self.loss_fn,
            weights=self.weights,
        )

        # try to load pregenerated points
        x = self._maybe_load_pregenerated(n)
        if x is None:
            x = self.train(args)  # (nbatch, n, d)

        # randomization
        if self.randomize == 'FALSE':
            if return_unrandomized:
                warnings.warn("return_unrandomized ignored when randomize='FALSE'.", stacklevel=1)
            return x[0] if self.nbatch == 1 else x

        # SHIFT
        xr = (x + self.shift[:, None, :]) % 1.0
        if self.nbatch == 1:
            xr0, x0 = xr[0], x[0]
            return (xr0, x0) if return_unrandomized else xr0
        return (xr, x) if return_unrandomized else xr

    def pdf(self, x):
        """ pdf of a standard uniform """
        return np.ones(x.shape[:-1], dtype=float)

    def __repr__(self):
        out = f"{self.__class__.__name__} Generator Object\n"
        for p in self.parameters:
            p_val = getattr(self, p)
            out += f"    {p:<15} {str(p_val)}\n"
        return out

    def _spawn(self, child_seed, dimension):
        """Spawn a child generator with same config (QMCPy hook)."""
        return MPMC(
            randomize=self.randomize,
            seed=child_seed,
            dimension=dimension,
            nbatch=self.nbatch,
            lr=self.lr,
            nlayers=self.nlayers,
            weight_decay=self.weight_decay,
            nhid=self.nhid,
            epochs=self.epochs,
            start_reduce=self.start_reduce,
            radius=self.radius,
            loss_fn=self.loss_fn,
            weights=self.weights.detach().cpu().tolist() if self.weights is not None else None,
        )

    # --------------------------
    # Helpers
    # --------------------------
    def _maybe_load_pregenerated(self, n):
        """Return (nbatch, n, d) array if pregenerated exists; else None."""
        head = "https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/pregenerated_pointsets/mpmc/"
        fname = f"dim_{self.dim}.nsamples_{n}.nbatch_{self.nbatch}.Lf{self.loss_fn}.b_{{b}}.txt"

        ds = np.lib.npyio.DataSource()
        # check batch 1; if absent, bail fast
        test_link = head + fname.format(b=1)
        if not ds.exists(test_link):
            return None

        batches = []
        for b in range(1, self.nbatch + 1):
            link = head + fname.format(b=b)
            if not ds.exists(link):
                warnings.warn(f"Missing pregenerated batch file: {link}; will train instead.", stacklevel=1)
                return None
            with ds.open(link) as fh:
                # files have 15-line headers
                data = np.loadtxt(fh, skiprows=15)
            if data.shape != (n, self.dim):
                warnings.warn(f"Pregenerated file shape {data.shape} mismatches (n,d)=({n},{self.dim}); will train instead.", stacklevel=1)
                return None
            batches.append(data)
        return np.asarray(batches, dtype=np.float64)

    # --------------------------
    # Training
    # --------------------------
    def train(self, args: SimpleNamespace):
        """
        Returns:
            np.ndarray: shape (nbatch, nsamples, dim)
        """
        model = MPMC_net(
            dim=args.dim, nhid=args.nhid, nlayers=args.nlayers,
            nsamples=args.nsamples, nbatch=args.nbatch,
            radius=args.radius, loss_fn=args.loss_fn, weights=args.weights
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float('inf')
        patience = 0
        end_result = None

        # adaptive schedule
        reduce_point = 10

        for epoch in tqdm(range(args.epochs),
                          desc=f"Training: N={args.nsamples}, d={args.dim}, loss={args.loss_fn}"):

            model.train()
            optimizer.zero_grad()
            loss, X = model()          # X: (nbatch, n, d)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                with torch.no_grad():
                    # compute batch discrepancies using the configured loss
                    fn = _DISCREPANCY[args.loss_fn]
                    if args.loss_fn.endswith('weighted'):
                        batched = fn(X, args.weights)
                    else:
                        batched = fn(X)
                    min_disc = batched.min().item()

                if min_disc < best_loss:
                    best_loss = min_disc
                    end_result = X.detach().cpu().numpy()

                # LR schedule after start_reduce
                if (epoch + 1) >= args.start_reduce:
                    if min_disc > best_loss:
                        patience += 1
                    if patience == reduce_point:
                        patience = 0
                        for g in optimizer.param_groups:
                            g['lr'] = max(g['lr'] / 10.0, 1e-6)
                        # stop early if lr already tiny
                        if optimizer.param_groups[0]['lr'] <= 1e-6:
                            break

        if end_result is None:
            end_result = X.detach().cpu().numpy()
        return end_result