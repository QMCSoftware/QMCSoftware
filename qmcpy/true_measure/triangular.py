import numpy as np

from ..util import ParameterError
from .scipy_wrapper import SciPyWrapper


class TriangularDistribution:
    """Triangular distribution matching scipy.stats.triang behavior.

    Support: [loc, loc + scale]
    Mode: loc + c*scale, with 0 < c < 1
    Provides ppf and pdf for SciPyWrapper custom-marginal usage.
    """

    def __init__(self, c=0.5, loc=0.0, scale=1.0) -> None:
        c = float(c)
        loc = float(loc)
        scale = float(scale)

        if not (0.0 < c < 1.0):
            raise ParameterError("c must lie strictly between 0 and 1.")
        if scale <= 0.0:
            raise ParameterError("scale must be positive.")

        self.c = c
        self.loc = loc
        self.scale = scale

        self._a = loc
        self._b = loc + scale
        self._m = loc + c * scale

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function of the triangular distribution.

        Args:
            x (np.ndarray): Points at which to evaluate the density.

        Returns:
            np.ndarray: Density values, same shape as `x`.
        """
        x = np.asarray(x, dtype=float)
        a, m, b = self._a, self._m, self._b
        out = np.zeros_like(x, dtype=float)

        left = (x >= a) & (x < m)
        right = (x >= m) & (x <= b)

        out[left] = 2.0 * (x[left] - a) / ((b - a) * (m - a))
        out[right] = 2.0 * (b - x[right]) / ((b - a) * (b - m))
        return out

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Percent point function (inverse CDF) of the triangular distribution.

        Args:
            u (np.ndarray): Probabilities in `[0,1]` at which to evaluate the inverse CDF.

        Returns:
            np.ndarray: Quantile values, same shape as `u`.
        """
        u = np.asarray(u, dtype=float)
        a, m, b = self._a, self._m, self._b
        Fm = (m - a) / (b - a)

        x = np.empty_like(u, dtype=float)
        left = u <= Fm
        right = ~left

        x[left] = a + np.sqrt(u[left] * (b - a) * (m - a))
        x[right] = b - np.sqrt((1.0 - u[right]) * (b - a) * (b - m))
        return x


class Triangular(SciPyWrapper):
    r"""Convenience TrueMeasure wrapper around TriangularDistribution.

    Examples:
        >>> from qmcpy import DigitalNetB2
        >>> tm = Triangular(DigitalNetB2(2, seed=7), c=0.25, loc=1.5, scale=3)
        >>> tm(4)
        array([[3.12921885, 3.74233664],
               [2.10644452, 2.53789087],
               [4.20108067, 1.77806228],
               [2.53774981, 2.77421122]])

        With independent replications:

        >>> tm_rep = Triangular(
        ...     DigitalNetB2(2, seed=7, replications=2),
        ...     c=0.25, loc=1.5, scale=3,
        ... )
        >>> tm_rep(4)
        array([[[2.90169471, 3.06638646],
                [1.85665708, 2.18758568],
                [3.51548593, 2.35181967],
                [2.31147618, 3.46328307]],
        <BLANKLINE>
               [[1.92173993, 2.26740964],
                [3.27010728, 3.13515076],
                [2.53166297, 3.86775554],
                [3.19663402, 1.81763309]]])
    """

    def __init__(self, sampler, c=0.5, loc=0.0, scale=1.0) -> None:
        super().__init__(
            sampler=sampler,
            scipy_distribs=TriangularDistribution(c=c, loc=loc, scale=scale),
        )
