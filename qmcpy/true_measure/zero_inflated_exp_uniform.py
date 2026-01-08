import numpy as np

from ..util import ParameterError, DimensionError
from .scipy_wrapper import SciPyWrapper

class _ZeroInflatedExpUniformAdapter:
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

        y[mask_zero] = self.y_split * u2[mask_zero]

        if np.any(mask_exp):
            u1r = (u1[mask_exp] - self.p_zero) / (1.0 - self.p_zero)
            x[mask_exp] = -np.log(1.0 - u1r) / self.lam
            y[mask_exp] = self.y_split + (1.0 - self.y_split) * u2[mask_exp]

        out = np.empty(u.shape, dtype=float)
        out[..., 0] = x
        out[..., 1] = y
        return out


class ZeroInflatedExpUniform(SciPyWrapper):
    def __init__(self, sampler, p_zero=0.4, lam=1.5, y_split=0.5):
        super().__init__(sampler=sampler, scipy_distribs=_ZeroInflatedExpUniformAdapter(p_zero, lam, y_split))
