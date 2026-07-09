import warnings

import numpy as np

from ..util import DimensionError, ParameterError
from .scipy_wrapper import SciPyWrapper


'''
class _ZeroInflatedExpUniform:
    def __init__(self, p_zero=0.4, lam=1.5):
        if not (0.0 < p_zero < 1.0):
            raise ParameterError("p_zero must be in (0,1).")
        if lam <= 0.0:
            raise ParameterError("lam must be positive.")

        self.p_zero = float(p_zero)
        self.lam = float(lam)
        self.dim = 1

    def transform(self, u):
        """
        Map unit-cube samples to a zero-inflated exponential distribution.

        The final axis stores distribution coordinates. For example:

            u.shape == (n, 1)       -> u[..., 0].shape == (n,)
            u.shape == (r, n, 1)    -> u[..., 0].shape == (r, n)

        where r is the number of replications and n is the number of samples.
        """
        u = np.asarray(u, dtype=float)      #(r, n, 1)

        if u.shape[-1] != 1:
            raise DimensionError(
                "ZeroInflatedExpUniform expects samples with shape (..., 1); "
                f"got {u.shape}."
            )

        # Select the only coordinate from every QMC point.
        # This preserves replication and sample axes.
        u = u[..., 0]  # (r, n)

        # Map u <= p_zero to the point mass at zero. 
        # For u > p_zero, rescale to (0,1) and apply the exponential inverse CDF.
        u = np.where(
            u <= self.p_zero,
            0.0,
            -np.log1p(
                -np.clip(
                    (u - self.p_zero) / (1.0 - self.p_zero),
                    np.finfo(float).eps,
                    1.0 - np.finfo(float).eps,
                )
            ) / self.lam,
        )

        # Restore the final distribution-coordinate axis.
        return u[..., None]      # (r, n, 1)


class ZeroInflatedExpUniform(SciPyWrapper):
    """
    One-dimensional zero-inflated exponential true measure.

    The ``y_split`` argument is retained temporarily for backward
    compatibility but is ignored.
    """

    def __init__(self, sampler, p_zero=0.4, lam=1.5, y_split=None):
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
            scipy_distribs=_ZeroInflatedExpUniform(
                p_zero=p_zero,
                lam=lam,
            ),
        )
'''

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