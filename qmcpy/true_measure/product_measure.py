import numpy as np

from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import DimensionError, ParameterError


class ProductMeasure(AbstractTrueMeasure):
    r"""
    Product true measure for independent composition of child true measures.

    The sampler must have dimension equal to the sum of the child dimensions.
    Unit-cube samples are split into child coordinate blocks, each child
    transform is applied to its own block, and transformed blocks are
    concatenated along the final axis.

    Exact product weights are supported for direct child true measures. For
    recursively composed child measures, sampling is supported through QMCPy's
    recursive transform helper, but exact final-space product weights are not
    currently implemented.
    """

    def __init__(self, sampler, children):
        if not isinstance(children, (list, tuple)) or len(children) == 0:
            raise ParameterError("ProductMeasure requires a nonempty list of children.")
        if not all(isinstance(child, AbstractTrueMeasure) for child in children):
            raise ParameterError(
                "Each ProductMeasure child must be an AbstractTrueMeasure instance."
            )
        if not isinstance(sampler, AbstractDiscreteDistribution):
            raise ParameterError(
                "ProductMeasure sampler must be an AbstractDiscreteDistribution."
            )

        self.parameters = ["children"]
        self.children = list(children)
        self.child_dimensions = np.array([child.d for child in self.children], dtype=int)
        self._split_indices = np.cumsum(self.child_dimensions)[:-1]
        self._total_child_dimension = int(self.child_dimensions.sum())
        if sampler.d != self._total_child_dimension:
            raise DimensionError(
                "ProductMeasure sampler dimension must equal the sum of child "
                f"dimensions ({sampler.d} != {self._total_child_dimension})."
            )

        self.domain = np.array([[0.0, 1.0]])
        self._parse_sampler(sampler)
        self.range = np.vstack(
            [
                self._expand_bounds(child.range, child.d, "range")
                for child in self.children
            ]
        )
        super(ProductMeasure, self).__init__()

    @staticmethod
    def _expand_bounds(bounds, dimension, name):
        bounds = np.asarray(bounds)
        if bounds.shape == (1, 2):
            return np.tile(bounds, (dimension, 1))
        if bounds.shape == (dimension, 2):
            return bounds
        raise DimensionError(
            f"Child true measure {name} must have shape (1, 2) or ({dimension}, 2)."
        )

    @property
    def _has_recursive_child(self):
        return any(child.transform != child for child in self.children)

    def _split_blocks(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.d:
            raise DimensionError(
                f"ProductMeasure expected last axis {self.d}, got {x.shape[-1]}."
            )
        return np.split(x, self._split_indices, axis=-1)

    def _transform(self, x):
        blocks = self._split_blocks(x)
        transformed_blocks = [
            child._jacobian_transform_r(block, return_weights=False)
            for child, block in zip(self.children, blocks)
        ]
        return np.concatenate(transformed_blocks, axis=-1)

    def _weight(self, x):
        if self._has_recursive_child:
            raise ParameterError(
                "ProductMeasure exact weights are currently supported only for "
                "direct child true measures."
            )

        blocks = self._split_blocks(x)
        weight = np.ones(np.asarray(x).shape[:-1], dtype=float)
        for child, block in zip(self.children, blocks):
            weight *= child._weight(block)
        return weight

    def _spawn(self, sampler, dimension):
        if dimension != self.d:
            raise DimensionError(
                "ProductMeasure spawning currently preserves the child dimensions."
            )
        return ProductMeasure(sampler=sampler, children=self.children)
