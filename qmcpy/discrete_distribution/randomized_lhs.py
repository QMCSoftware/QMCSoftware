from qmcpy.discrete_distribution.abstract_discrete_distribution import AbstractDiscreteDistribution
import numpy as np
from qmcpy.util import ParameterError, ParameterWarning
import warnings


class RandomizedLHS(AbstractDiscreteDistribution):
    r"""
    Latin Hypercube Sampler for quasi-Monte Carlo and experimental design.

    Latin Hypercube Sampling (LHS) generates points with excellent univariate
    stratification: splitting $[0,1)$ into `n` equal strata along *any* single
    coordinate axis places exactly one point in each stratum. Introduced by
    McKay, Beckman, and Conover as a variance-reduction alternative to simple
    random sampling for computer experiments, LHS is asymptotically at least
    as accurate as Monte Carlo for the additive part of an integrand, with the
    rate of improvement characterized by Stein and later by Loh via a
    multivariate central limit theorem.

    Note:
        - Unlike the low discrepancy sequences in this package (e.g. `Lattice`,
          `Halton`, `DigitalNetB2`), `RandomizedLHS` points are *not* extensible
          in `n`: the entire point set must be regenerated whenever `n` changes,
          since the strata boundaries themselves depend on `n`. Consequently
          `n_min` and `n_max` only matter through their difference `n_max`-`n_min`;
          the absolute indices carry no meaning.
        - `replications` produces independent randomizations (independent random
          permutations and independent within-stratum jitter), not an extension
          or reshaping of a single sequence.

    Examples:
        >>> discrete_distrib = RandomizedLHS(2,replications=None,seed=7)
        >>> discrete_distrib(4)
        array([[0.10079093, 0.46583043],
               [0.73559373, 0.81194835],
               [0.44928008, 0.53874559],
               [0.9427258 , 0.10932748]])

        Replications of independent randomizations 

        >>> x = RandomizedLHS(3,replications=2,seed=7)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.10407119, 0.26180415, 0.40801808],
                [0.599802  , 0.86538519, 0.71594779],
                [0.41110958, 0.60764889, 0.84352474],
                [0.91828595, 0.20655501, 0.12138689]],
        <BLANKLINE>
               [[0.12951706, 0.17802722, 0.63472096],
                [0.9652066 , 0.31881758, 0.78031776],
                [0.3523449 , 0.55568244, 0.26354665],
                [0.74188086, 0.98836479, 0.12774161]]])

    Parameters
    ----------
    dimension (int): Dimension of the samples.

    replications (Union[None, int]): Number of independent randomizations. This is implemented only for API consistency; equivalent to reshaping the samples.

    seed (Union[None, int, np.random.SeedSequence]): Seed for the random number generator to ensure reproducibility.

    **References:**

    1.  M. D. McKay, R. J. Beckman, and W. J. Conover.  
        A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code.  
        Technometrics, 21(2):239-245, 1979.  
        [https://doi.org/10.1080/00401706.1979.10489755](https://doi.org/10.1080/00401706.1979.10489755).

    2.  M. Stein.  
        Large Sample Properties of Simulations Using Latin Hypercube Sampling.  
        Technometrics, 29(2):143-151, 1987.  
        [https://doi.org/10.1080/00401706.1987.10488205](https://doi.org/10.1080/00401706.1987.10488205).

    3.  A. B. Owen.  
        Controlling Correlations in Latin Hypercube Samples.  
        Journal of the American Statistical Association, 89(428):1517-1522, 1994.  
        [https://doi.org/10.1080/01621459.1994.10476891](https://doi.org/10.1080/01621459.1994.10476891).

    4.  W.-L. Loh.  
        On Latin Hypercube Sampling.  
        The Annals of Statistics, 24(5):2058-2080, 1996.  
        [https://doi.org/10.1214/aos/1069362310](https://doi.org/10.1214/aos/1069362310).

    5.  B. Tang.  
        Orthogonal Array-Based Latin Hypercubes.  
        Journal of the American Statistical Association, 88(424):1392-1397, 1993.  
        [https://doi.org/10.1080/01621459.1993.10476423](https://doi.org/10.1080/01621459.1993.10476423).
    """

    def __init__(
            self,dimension ,replications ,seed
            ) :
        super().__init__(dimension = dimension, replications = replications, seed = seed , d_limit = np.inf, n_limit = np.inf)

    def _gen_samples(
            self, n=None, n_min=None, n_max=None, return_binary=False, warn=True
        ) :
        r"""
        Generate Latin Hypercube samples in the unit hypercube [0, 1)^dimension.

        - If only `n` is supplied, generate samples from the sequence at indices 0,...,`n`-1.
        - If `n_min` and `n_max` are supplied, generate samples from the sequence at indices `n_min`,...,`n_max`-1.
        - If `n` and `n_min` are supplied, generate samples from the sequence at indices `n`,...,`n_min`-1.

        Args:
            n (Union[None, int]): Number of points to generate.
            n_min (Union[None, int]): Starting index of the sequence.
            n_max (Union[None, int]): Final index of the sequence.
            warn (bool): If `False`, disable warnings when generating samples.

        Returns:
            x (np.ndarray): Samples from the sequence.

                - If `replications` is `None`, this will be of size (`n_max`-`n_min`) $\times$ `dimension`.
                - If `replications` is a positive int, `x` will be of size `replications` $\times$ (`n_max`-`n_min`) $\times$ `dimension`.
        """
        if return_binary:
                raise ParameterError("RandomizedLHS does not support return_binary=True")
        if warn:
                warnings.warn(
                        "For RandomizedLHS , the values of n_max and n_min are not really important. The only thing that matters is the total number of points 'n' "
                        "in order to split each dimension correctly and to place each coordinate of each point within its own stratum",
                        ParameterWarning,
                    )
        n = int(n_max - n_min)
        keys = self.rng.random(size=(self.replications,self.d,n)) # random numbers 
        perm_indices = np.argsort(keys, axis=-1) 
        permutations = perm_indices + 1
        U = self.rng.uniform(0,1 , size = permutations.shape)
        result = ((permutations - U) / n)
        result = result.transpose(0, 2, 1) 
        return result

    def _spawn(self, child_seed, dimension):
            return RandomizedLHS(
                dimension=dimension,
                replications=None if self.no_replications else self.replications,
                seed=child_seed,
            )
    def __repr__(self):
        return super().__repr__("RandomizedLHS")