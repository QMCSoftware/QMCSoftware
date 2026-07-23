import warnings

import numpy as np

from ..util import DimensionError, ParameterError
from .scipy_wrapper import SciPyWrapper


class _ZeroInflatedExponential:
    """
    One-dimensional zero-inflated exponential distribution.

    This distribution has probability mass ``p_zero`` at zero and an
    exponential distribution with rate ``lam`` on positive values.

    It implements ``ppf`` so it can be passed to ``SciPyWrapper`` as a
    custom univariate marginal.
    """

    def __init__(self, p_zero=0.4, lam=1.5):
        if not (0.0 < p_zero < 1.0):
            raise ParameterError("p_zero must be in (0,1).")
        if lam <= 0.0:
            raise ParameterError("lam must be positive.")

        self.p_zero = float(p_zero)
        self.lam = float(lam)

    def ppf(self, u):
        """
        Generalized inverse CDF of the zero-inflated exponential.

        SciPyWrapper supplies one coordinate at a time. For example:

            sampler output: (n, 1)
            ppf input:      (n,)
        """
        u = np.asarray(u, dtype=float)

        # Values up to p_zero map to the point mass at X = 0.
        x = np.zeros_like(u, dtype=float)
        mask_exp = u > self.p_zero

        # Rescale the remaining values to (0, 1), then use the
        # exponential inverse CDF.
        if np.any(mask_exp):
            u_rescaled = (u[mask_exp] - self.p_zero) / (
                1.0 - self.p_zero
            )
            u_rescaled = np.clip(
                u_rescaled,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
            x[mask_exp] = -np.log1p(-u_rescaled) / self.lam

        return x


class ZeroInflatedExpUniform(SciPyWrapper):
    """
    One-dimensional zero-inflated exponential true measure.

    The ``y_split`` keyword is retained temporarily for backward
    compatibility but is ignored.

    Examples
    --------
    Without replications:

    >>> from qmcpy.discrete_distribution import DigitalNetB2
    >>> from qmcpy.true_measure import ZeroInflatedExpUniform
    >>> tm = ZeroInflatedExpUniform(
    ...     DigitalNetB2(1, seed=7), p_zero=0.4, lam=1.5
    ... )
    >>> x = tm(8)
    >>> x
    array([[0.        ],
           [0.76621559],
           [0.        ],
           [0.18405583],
           [0.08112272],
           [1.19997153],
           [0.        ],
           [0.33259467]])
    >>> x.shape
    (8, 1)
    >>> bool((x >= 0).all())
    True

    With independent replications:

    >>> tm = ZeroInflatedExpUniform(
    ...     DigitalNetB2(1, seed=7, replications=2),
    ...     p_zero=0.4,
    ...     lam=1.5,
    ... )
    >>> x = tm(8)
    >>> x
    array([[[0.51197024],
            [0.        ],
            [2.54258665],
            [0.03368876],
            [0.2192598 ],
            [0.        ],
            [0.85384192],
            [0.        ]],
    <BLANKLINE>
           [[1.3024994 ],
            [0.03378461],
            [0.20489897],
            [0.        ],
            [0.58638285],
            [0.        ],
            [0.35227285],
            [0.        ]]])
    >>> x.shape
    (2, 8, 1)
    >>> bool((x >= 0).all())
    True
    """

    def __init__(self, sampler, p_zero=0.4, lam=1.5, y_split=None):
        if sampler.d != 1:
            raise DimensionError(
                "ZeroInflatedExpUniform requires a one-dimensional sampler."
            )

        if y_split is not None:
            warnings.warn(
                "`y_split` is deprecated and ignored. "
                "`ZeroInflatedExpUniform` is now one-dimensional; "
                "remove `y_split` from this call.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(
            sampler=sampler,
            scipy_distribs=_ZeroInflatedExponential(
                p_zero=p_zero,
                lam=lam,
            ),
        )
