import numpy as np

from ..util import DimensionError, ParameterError
from .scipy_wrapper import SciPyWrapper

from ..discrete_distribution import DigitalNetB2


class _ZeroInflatedExpUniformAdapter:
    """
    Joint distribution adapter with transform(u) only.

    With prob p_zero:
        X = 0
        Y ~ Uniform(0, y_split)

    With prob 1 - p_zero:
        X ~ Exp(lam)
        Y ~ Uniform(y_split, 1)
    """

    def __init__(self, p_zero=0.4, lam=1.5, y_split=0.5):
        if not (0.0 < p_zero < 1.0):
            raise ParameterError("p_zero must be in (0,1).")
        if lam <= 0.0:
            raise ParameterError("lam must be positive.")
        if not (0.0 < y_split < 1.0):
            raise ParameterError("y_split must be in (0,1).")

        self.p_zero = float(p_zero)
        self.lam = float(lam)
        self.y_split = float(y_split)
        self.dim = 2

    def transform(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != 2:
            raise DimensionError(f"Expected last axis 2, got {u.shape[-1]}")

        u1 = u[..., 0]
        u2 = u[..., 1]

        x = np.zeros_like(u1, dtype=float)
        y = np.empty_like(u1, dtype=float)

        mask_zero = u1 <= self.p_zero
        mask_exp = ~mask_zero

        # Zero branch
        y[mask_zero] = self.y_split * u2[mask_zero]

        # Exp branch
        if np.any(mask_exp):
            u1_rescaled = (u1[mask_exp] - self.p_zero) / (1.0 - self.p_zero)
            x[mask_exp] = -np.log(1.0 - u1_rescaled) / self.lam
            y[mask_exp] = self.y_split + (1.0 - self.y_split) * u2[mask_exp]

        out = np.empty(u.shape, dtype=float)
        out[..., 0] = x
        out[..., 1] = y
        return out


class ZeroInflatedExpUniformJoint(SciPyWrapper):
    """
    Convenience true measure for the zero-inflated joint example.

    Notes:
    - This joint has no logpdf, so SciPyWrapper will treat weights as 1.

    Example:
    >>> tm = ZeroInflatedExpUniformJoint(
    ...     sampler=DigitalNetB2(2, seed=7),
    ...     p_zero=0.4,
    ...     lam=1.5,
    ...     y_split=0.5,
    ... )
    >>> tm(4).shape
    (4, 2)
    """

    def __init__(self, sampler, p_zero=0.4, lam=1.5, y_split=0.5):
        adapter = _ZeroInflatedExpUniformAdapter(p_zero=p_zero, lam=lam, y_split=y_split)
        super().__init__(sampler=sampler, scipy_distribs=adapter)
