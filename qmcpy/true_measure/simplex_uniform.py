import math
import warnings
from typing import Union

import numpy as np

from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..discrete_distribution import DigitalNetB2
from .simplex_transform import _SimplexTransform
from ..util import ParameterError
from .abstract_true_measure import AbstractTrueMeasure


class SimplexUniform(AbstractTrueMeasure):
    r"""Uniform distribution on the corner simplex $K_d = \{w \in \mathbb{R}^d :
    w_i \ge 0, \sum_{i=1}^d w_i \le 1\}$, named for its shape: the corner of
    the unit cube $[0,1]^d$ cut off by the hyperplane $\sum_i w_i = 1$.

    Returns $d$ of $d+1$ nonnegative weights summing to 1; append $w_{d+1} =
    1 - \sum_i w_i$ for the full probability-simplex vector, which is
    $\mathrm{Dirichlet}(1,\dots,1)$-distributed ($d+1$ ones) [1].

    Wraps `_SimplexTransform`'s mapping onto the *ordered* simplex $T_d = \{0
    \le x_1 \le \dots \le x_d \le 1\}$, then takes consecutive differences
    ($w_i = x_i - x_{i-1}$, $x_0 := 0$), a volume-preserving (Jacobian 1)
    reparametrization, so the density is $d!$ either way.

    References:
        [1] L. Devroye, "Non-Uniform Random Variate Generation," Springer-Verlag,
        1986, Sec. I.4.1 (sampling the simplex via uniform spacings). [Online].
        Available: http://luc.devroye.org/rnbookindex.html

    Examples:
        >>> s = SimplexUniform(DigitalNetB2(3, seed=7))
        >>> w = s(4)
        >>> w.shape
        (4, 3)
        >>> bool(np.all(w >= 0) and np.all(w.sum(axis=-1) <= 1))
        True

        Each coordinate's moments follow the symmetric Dirichlet$(1,\dots,1)$
        marginal/pairwise formulas $E[W_i] = \frac{1}{d+1}$,
        $\mathrm{Var}(W_i) = \frac{d}{(d+1)^2(d+2)}$,
        $\mathrm{Cov}(W_i,W_j) = -\frac{1}{(d+1)^2(d+2)}$ ($i \ne j$). Unlike
        `Kumaraswamy`'s independent coordinates, this covariance is dense, not
        diagonal: the $d+1$ weights share a fixed budget summing to 1.

        >>> s2 = SimplexUniform(DigitalNetB2(2, seed=7))
        >>> s2  # doctest: +NORMALIZE_WHITESPACE
        SimplexUniform (AbstractTrueMeasure)
            transform_method root
            mean            [0.333 0.333]
            variance        [0.056 0.056]
            standard_deviation [0.236 0.236]
            covariance      [[ 0.056 -0.028]
                             [-0.028  0.056]]

        Every `transform_method` is measure-preserving (weight means over
        500,000 samples matched the exact Dirichlet mean $1/(d+1)$), so the
        choice is a QMC-efficiency question, not a correctness one:

        >>> s = SimplexUniform(DigitalNetB2(3, seed=7), transform_method='shift')
        >>> s(4).shape
        (4, 3)

        Append the implicit $(d+1)$-th weight for the full probability-simplex
        vector (sums to exactly 1):

        >>> w = s(1)[0]
        >>> v = np.append(w, 1 - w.sum())
        >>> v.shape
        (4,)
        >>> bool(np.isclose(v.sum(), 1) and np.all(v >= 0))
        True
    """

    def __init__(self, sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure],
                 transform_method: str = "root") -> None:
        """Initialize a SimplexUniform true measure.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.

                Its dimension must be <= 170: the density d! exceeds float64's
                range above that, raising ParameterError at construction.
            transform_method (str): One of _SimplexTransform's measure-preserving
                methods: 'root', 'sort', 'shift', 'origami', or 'mirror'.
                'mirror' folds a symmetric point set (e.g. a lattice) onto
                itself, degrading its low-discrepancy structure, and only
                supports dimension <= 3 (raised by _SimplexTransform itself);
                a warning is issued when it is selected. ('drop' is excluded
                entirely: it rejects rather than maps points 1:1, so it does
                not fit the _transform/_weight contract at all; see
                AcceptanceRejection for that pattern instead.)
        """
        if transform_method not in ("root", "sort", "shift", "origami", "mirror"):
            raise ParameterError(
                "transform_method must be one of 'root', 'sort', 'shift', 'origami', "
                f"'mirror', not {transform_method!r}"
            )
        if transform_method == "mirror":
            warnings.warn(
                "transform_method='mirror' folds a symmetric point set (e.g., a "
                "lattice) onto itself, which can degrade its low-discrepancy "
                "structure, and only supports dimension <= 3.",
                UserWarning,
            )
        self.transform_method = transform_method
        self.parameters = ["transform_method", "mean", "variance", "standard_deviation", "covariance"]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self._simplex = _SimplexTransform(dimension=self.d)
        try:
            self._density = float(math.factorial(self.d))  # 1 / Vol(T_d), Vol(T_d) = 1/d!
        except OverflowError:
            raise ParameterError(
                f"SimplexUniform's density d! is not representable as a float64 "
                f"for dimension {self.d} (float64 overflows above d=170)."
            )
        self.range = np.tile(np.array([0, 1]), (self.d, 1))
        # The d returned weights are the first d coordinates of a symmetric
        # Dirichlet(1,...,1) vector with d+1 parts: E[W_i] = 1/(d+1),
        # Var(W_i) = d/((d+1)^2 (d+2)), Cov(W_i,W_j) = -1/((d+1)^2 (d+2)) for i != j.
        d = self.d
        var_val = d / ((d + 1) ** 2 * (d + 2))
        cov_val = -1.0 / ((d + 1) ** 2 * (d + 2))
        covariance = np.full((d, d), cov_val)
        np.fill_diagonal(covariance, var_val)
        self._set_moments(
            mean=np.full(d, 1.0 / (d + 1)),
            variance=np.full(d, var_val),
            standard_deviation=np.full(d, math.sqrt(var_val)),
            covariance=covariance,
        )
        super(SimplexUniform, self).__init__()

    def _transform(self, x):
        t = getattr(self._simplex, self.transform_method)(x)
        # t is an ordered-simplex T_d point (d sorted coords); consecutive differences
        # (x_0 := 0) give the first d of d+1 weights, a Jacobian-1 map that leaves density unchanged.
        return np.diff(t, axis=-1, prepend=0)

    @staticmethod
    def transform_points(x: np.ndarray, transform_method: str = "root") -> np.ndarray:
        r"""Map pre-generated points through a simplex transform directly,
        without constructing a `SimplexUniform` or generating samples via
        `gen_samples`/`__call__`.

        Has no dimension limit, unlike the class itself: it never computes
        this measure's density ($d!$, which overflows float64 above $d=170$
        and is only needed by `_weight`/a weighted `gen_samples`), so it stays
        usable for an externally-shared point set of any width.

        For sharing one point set between several constructions on identical
        draws (e.g. comparing `transform_method`s, or feeding a point set
        wider than what a single `SimplexUniform` would consume), generate
        points once from any sampler, then pass slices of them here instead
        of letting each construction draw its own.

        Args:
            x (np.ndarray): Points with shape `(..., d)`, each row in
                $[0,1]^d$. Not validated against any sampler; the caller is
                responsible for `x` being an appropriate input (e.g. a
                uniform point set) for `transform_method`.
            transform_method (str): One of `_SimplexTransform`'s measure-
                preserving methods: 'root', 'sort', 'shift', 'origami', or
                'mirror' (same constraints as `__init__`'s own Args).

        Returns:
            np.ndarray: `d` of the `d+1` corner-simplex weights, shape
                `(..., d)`; append the implicit `(d+1)`-th weight for the
                full vector, as in the class docstring's last example.

            >>> x = DigitalNetB2(3, seed=7).gen_samples(4)
            >>> w = SimplexUniform.transform_points(x)
            >>> w.shape
            (4, 3)
            >>> full = np.concatenate([w, 1 - w.sum(axis=-1, keepdims=True)], axis=-1)
            >>> bool(np.allclose(full.sum(axis=-1), 1) and np.all(full >= 0))
            True
        """
        if transform_method not in ("root", "sort", "shift", "origami", "mirror"):
            raise ParameterError(
                "transform_method must be one of 'root', 'sort', 'shift', 'origami', "
                f"'mirror', not {transform_method!r}"
            )
        if transform_method == "mirror":
            warnings.warn(
                "transform_method='mirror' folds a symmetric point set (e.g., a "
                "lattice) onto itself, which can degrade its low-discrepancy "
                "structure, and only supports dimension <= 3.",
                UserWarning,
            )
        t = getattr(_SimplexTransform(dimension=x.shape[-1]), transform_method)(x)
        return np.diff(t, axis=-1, prepend=0)

    def _weight(self, x):
        eps = np.finfo(float).eps
        inside = np.all(x >= -eps, axis=-1) & (x.sum(axis=-1) <= 1+eps)
        return np.where(inside, self._density, 0.0)

    def _spawn(self, sampler, dimension):
        return SimplexUniform(sampler, transform_method=self.transform_method)
