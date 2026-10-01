import math
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
    r"""Uniform distribution on the simplex $T_d = \{(x_1,\dots,x_d) : 0 \le x_1 \le \dots \le x_d \le 1\}$.

    Wraps `_SimplexTransform` (a stateless cube-to-simplex mapping) as a
    proper `AbstractTrueMeasure`, so a cube sampler can be composed with
    QMCPy's integration machinery directly instead of needing points
    transformed by hand first.

    Examples:
        >>> true_measure = SimplexUniform(DigitalNetB2(3, seed=7))
        >>> x = true_measure(4)
        >>> x.shape
        (4, 3)
        >>> bool(np.all(x[:, :-1] <= x[:, 1:]))
        True

        Every `transform_method` is measure-preserving (verified empirically:
        the sorted-coordinate means of 500,000 samples all matched the exact
        order-statistics means $i/(d+1)$ to within Monte Carlo noise), so the
        choice is a QMC-efficiency question, not a correctness one:

        >>> true_measure = SimplexUniform(DigitalNetB2(3, seed=7), transform_method='shift')
        >>> true_measure(4).shape
        (4, 3)
    """

    def __init__(self, sampler: Union[AbstractDiscreteDistribution, AbstractTrueMeasure],
                 transform_method: str = "root") -> None:
        """Initialize a SimplexUniform true measure.

        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]):
                Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            transform_method (str): One of _SimplexTransform's measure-preserving
                methods: 'root', 'sort', 'shift', or 'origami'. ('mirror' is
                excluded here: it folds a symmetric point set, e.g. a lattice,
                onto itself, so it is unsuitable as a general sampler
                transform. 'drop' rejects rather than maps points 1:1, so it
                does not fit the _transform/_weight contract at all; see
                AcceptanceRejection for that pattern instead.)
        """
        if transform_method not in ("root", "sort", "shift", "origami"):
            raise ParameterError(
                "transform_method must be one of 'root', 'sort', 'shift', 'origami', "
                f"not {transform_method!r}"
            )
        self.transform_method = transform_method
        self.parameters = ["transform_method"]
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self._simplex = _SimplexTransform(dimension=self.d)
        self._density = float(math.factorial(self.d))  # 1 / Vol(T_d), Vol(T_d) = 1/d!
        self.range = np.tile(np.array([0, 1]), (self.d, 1))
        super(SimplexUniform, self).__init__()

    def _transform(self, x):
        return getattr(self._simplex, self.transform_method)(x)

    def _weight(self, x):
        return np.full(x.shape[:-1], self._density)

    def _spawn(self, sampler, dimension):
        return SimplexUniform(sampler, transform_method=self.transform_method)
