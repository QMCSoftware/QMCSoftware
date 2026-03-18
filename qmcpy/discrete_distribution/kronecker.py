from .abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ..util import ParameterError
import numpy as np
import warnings
from sympy import nextprime

PRIMES = np.array([2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41, 
                   43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97, 101,
                   103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
                   173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
                   241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
                   317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
                   401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
                   479, 487, 491, 499, 503, 509, 521, 523, 541])

RICHTMYER = np.sqrt(PRIMES) % 1
SUZUKI = lambda d: 2 ** (np.arange(1, d + 1) / (d + 1))
CBC = np.array([4.224371872086318813e-01,
3.605965189622313272e-01,
3.486721371284548510e-01,
4.520388055082059653e-01,
2.550750763977845947e-01,
2.205289926147350477e-01,
2.071242872822959824e-01,
3.049354991086913325e-01,
3.872168854974577523e-01,
2.275808872220986823e-01,
1.773740893160189180e-01,
1.958399682530986008e-01,
3.216346950830996643e-01], dtype=np.float64)

class Kronecker(AbstractLDDiscreteDistribution):
    r"""
    Kronecker sequence (additive recurrence sequence) for quasi-Monte Carlo.

    A Kronecker sequence is defined by
    $$
    \bolsymbol{x}_i =  i \boldsymbol{\alpha} + \boldsymbol{\delta} \bmod \boldsymbol{1} \in [0,1)^d, \quad i = 0,1,2,\dots,
    $$
    where $\boldsymbol{\alpha} \in \mathbb{R}^d$ is a generating vector and $\boldsymbol{\delta} \in [0,1)^d$
    is an optional shift. The fractional part is taken componentwise.

    These sequences are simple, extensible low-discrepancy sequences when
    $\boldsymbol{\alpha}$ has components that are irrational and well-distributed.

    Parameters:
        dimension (int):
            Dimension $d$ of the sequence.

        alpha (str or array-like):
            Generating vector $\boldsymbol{\alpha}$.
            
            Options:
            - "RICHTMYER": uses $\boldsymbol{\alpha}_j = \sqrt{p_j} \bmod 1$, where $p_j$ are primes.
              This is the classical Richtmyer construction.
            - "SUZUKI": uses a deterministic construction
              $\boldsymbol{\alpha}_j = 2^{j/(d+1)}$.
            - array-like: user-specified generating vector.

        delta (array-like, optional):
            Shift vector $\boldsymbol{\delta}$. If `randomize=True`, this is ignored and
            a random shift is generated. Otherwise, a fixed shift is used.

        replications (int, optional):
            Number of independent randomizations (replications).

        randomize (bool):
            If True, apply a random shift $\boldsymbol{\delta} \sim \mathrm{Uniform}([0,1)^d)$.
            If False, use the provided `delta` or zero shift.

        seed (int, optional):
            Random seed for reproducibility.

    Notes:
        - The Kronecker sequence is fully extensible in $n$ (no restriction to powers of 2).
        - Quality depends strongly on the choice of $\boldsymbol{\alpha}$.
        - Random shifting preserves unbiasedness for integration.

    Examples:
        >>> dd = Kronecker(dimension=2, seed=7)
        >>> dd(4)
        array([[...]])

        First point (n = 0):
        >>> dd(1)
        array([[...]])

        Multiple replications:
        >>> x = Kronecker(3, seed=7, replications=2)(4)
        >>> x.shape
        (2, 4, 3)

    References:
        - Richtmyer, R. D. (1951). "The evaluation of definite integrals and a quasi-Monte Carlo method."
        - Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods*.
    """
        
    def __init__(self, dimension=1, alpha="CBC", delta=None, replications=None, randomize=True, seed=None):
        self.parameters = ["randomize", "alpha", "n_limit"]
        self.input_alpha = alpha
        # attributes required for cub_qmc_clt.py
        self.mimics = 'StdUniform'
        self.randomize = randomize
                
        if type(alpha) == str and alpha.lower() == 'cbc':
            if dimension <= len(CBC):
                self.alpha = CBC[:dimension]
            else:
                warnings.warn(
                    f"CBC generating vector only supports dimension <= {len(CBC)}; "
                    "falling back to Richtmyer.",
                    RuntimeWarning
                )
                if dimension <= len(PRIMES):
                    self.alpha = RICHTMYER[:dimension]
                else:
                    self.alpha = np.append(
                        RICHTMYER,
                        [np.sqrt(nextprime(PRIMES[-1], ith=x)) % 1 for x in range(1, dimension - len(PRIMES) + 1)]
                    )        
        elif type(alpha) == str and alpha.lower() == 'richtmyer':
                if dimension <= len(PRIMES):
                    self.alpha = RICHTMYER[:dimension]
                else:
                    self.alpha = np.append(RICHTMYER, [np.sqrt(nextprime(PRIMES[-1], ith=x)) % 1 for x in range(1, dimension - len(PRIMES) + 1)])
        elif type(alpha) == str and alpha.lower() == 'suzuki':
            self.alpha = SUZUKI(dimension)
        else:
            alpha = np.asarray(alpha, dtype=float)
            if alpha.ndim != 1:
                raise ParameterError("alpha must be a 1D array-like")
            if len(alpha) < dimension:
                raise ParameterError(
                    f"alpha length {len(alpha)} is less than dimension {dimension}"
                )
            self.alpha = alpha[:dimension]

        super().__init__(dimension,replications,seed,d_limit=np.inf,n_limit=np.inf) 

        # validate delta first if provided
        if delta is not None:
            delta = np.asarray(delta, dtype=float)
            if delta.ndim == 1:
                if len(delta) != self.d:
                    raise ParameterError("delta must have length equal to dimension")
                delta = np.tile(delta, (self.replications, 1))
            elif delta.ndim == 2:
                if delta.shape != (self.replications, self.d):
                    raise ParameterError(
                        "delta must have shape (replications, dimension)"
                    )
            else:
                raise ParameterError("delta must be 1D or 2D array-like")

        # now apply randomization logic
        if self.randomize:
            self.delta = self.rng.uniform(size=(self.replications, self.d))
        elif delta is not None:
            self.delta = delta
        else:
            self.delta = np.zeros((self.replications, self.d))

    def _gen_samples(self, n_min, n_max, return_binary, warn):
        # returns replications x (n_max-n_min) x d (dimension) array of samples

        i = np.arange(n_min,n_max).reshape((n_max-n_min, 1))

        points = ((i * self.alpha) + self.delta[:,None,:]) % 1

        return points


    def periodic_discrepancy(self, n, k_tilde=None, gamma=None):
        """
        Calculates the discrepancy for a periodic kernel.

        Args:
            n (int): the number of sample points
            k_tilde (tuple(function, float)): the function takes in 2 arguments: the sample points and the coordinate weights.
                The float is the integral over the unit hypercube.
            gamme (ndarray): shape (1xd)

        Returns:
            float
        
        Note:
            If k_tilde is not specified, the second Bernoulli polynomial is used.
            If gamma is not specified, the coordinate weights will be just all ones.
        """
        if gamma is None:
            gamma = np.ones(self.d)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: np.prod(1 + (x * (x - 1) + 1/6) * gamma, axis=-1), 1)

        return np.sqrt(self._square_periodic_discrepancies(n, k_tilde, gamma))
        

    # calculates the weighted sum of square discrepancy
    def wssd_discrepancy(self, n, weights, k_tilde = None, gamma = None):
        if gamma is None:
            gamma = np.ones(self.d)

        if k_tilde is None:
            k_tilde = (lambda x, gamma: np.prod(1 + (x * (x - 1) + 1/6) * gamma, axis=-1), 1)

        discrepancies = self._square_periodic_discrepancies(n, k_tilde, gamma)
        return np.sum(weights * discrepancies, axis=-1)

    
    def _square_periodic_discrepancies(self, n, k_tilde, gamma):
        n_array = np.arange(1, n + 1)
        k_tilde_terms = k_tilde[0](self.gen_samples(n=n), gamma)

        left_sum = np.cumsum(k_tilde_terms[...,1:], axis=-1) * n_array[1:]
        right_sum = np.cumsum(n_array[:-1] * k_tilde_terms[...,1:], axis=-1)

        k_tilde_zero_terms = k_tilde_terms[...,0] * n_array
        summation = np.zeros_like(k_tilde_terms)
        summation[...,1:] = left_sum - right_sum
        return (k_tilde_zero_terms + 2 * summation) / (n_array ** 2) - k_tilde[1]
    
    
    def _spawn(self, child_seed, dimension):
        return Kronecker(
                dimension=dimension,
                alpha = self.input_alpha,
                randomize=self.randomize,
                seed=child_seed,
                replications=self.replications)