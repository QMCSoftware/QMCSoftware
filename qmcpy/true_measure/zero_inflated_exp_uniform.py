import warnings

import numpy as np

from ..util import DimensionError, ParameterError
from .scipy_wrapper import SciPyWrapper


class _ZeroInflatedExpUniformAdapter:
    def __init__(self, p_zero=0.4, lam=1.5):
        if not (0.0 < p_zero < 1.0):
            raise ParameterError("p_zero must be in (0,1).")
        if lam <= 0.0:
            raise ParameterError("lam must be positive.")

        self.p_zero = float(p_zero)
        self.lam = float(lam)
        self.dim = 1

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != 1:
            raise DimensionError(f"Expected last axis 1, got {u.shape[-1]}")

        u1 = u[..., 0]
        x = np.zeros_like(u1, dtype=float)
<<<<<<< Updated upstream

        mask_exp = u1 > self.p_zero
=======
>>>>>>> Stashed changes

        mask_exp = u1 > self.p_zero
        if np.any(mask_exp):
            u1r = (u1[mask_exp] - self.p_zero) / (1.0 - self.p_zero)
            u1r = np.clip(u1r, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
            x[mask_exp] = -np.log1p(-u1r) / self.lam

        return x[..., None]


class ZeroInflatedExpUniform(SciPyWrapper):
<<<<<<< Updated upstream
    def __init__(self, sampler, p_zero=0.4, lam=1.5):
        super().__init__(
            sampler=sampler,
            scipy_distribs=_ZeroInflatedExpUniformAdapter(p_zero, lam),
=======
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
            scipy_distribs=_ZeroInflatedExpUniformAdapter(
                p_zero=p_zero,
                lam=lam,
            ),
>>>>>>> Stashed changes
        )