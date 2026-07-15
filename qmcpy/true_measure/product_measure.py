import numpy as np

from .abstract_true_measure import AbstractTrueMeasure
from ..discrete_distribution.abstract_discrete_distribution import (
    AbstractDiscreteDistribution,
)
from ..util import DimensionError, ParameterError


class ProductMeasure(AbstractTrueMeasure):
    r"""
    Product true measure for independent composition of child true measures.

    `ProductMeasure` represents an independent product of smaller true
    measures. If the child true measures have dimensions

    d_1, d_2, ..., d_k,

    then the product measure has total dimension

    d = d_1 + d_2 + ... + d_k.

    A single d-dimensional sampler is used. Its unit-cube samples are split
    into coordinate blocks, one block for each child true measure. Each child
    transforms its own block, and the transformed blocks are concatenated
    back together.

    For example, if the children are

        child 1: 2D Gaussian
        child 2: 1D zero-inflated exponential

    then ProductMeasure uses a 3D sampler and returns samples with three
    coordinates. The first two coordinates come from the Gaussian block, and
    the third coordinate comes from the zero-inflated exponential block.

    The child true measures still have their own samplers because QMCPy's
    current ``AbstractTrueMeasure`` API requires every true measure to be
    constructed with one. Inside ``ProductMeasure``, those child samplers are
    construction placeholders: they are not sampled directly when product
    samples are generated. The children provide dimension, range, transform,
    and weight behavior.

    Notes
    -----
    Exact product weights are supported for direct child true measures. For
    recursively composed child measures, sampling is supported through
    QMCPy's recursive transform helper, but exact final-space product weights
    are not currently implemented here.

    Examples
    --------
    Combine two one-dimensional uniform true measures:

    >>> from qmcpy.discrete_distribution import DigitalNetB2
    >>> from qmcpy.true_measure import ProductMeasure, Uniform
    >>> def dummy_child_sampler(d):
    ...     return DigitalNetB2(d, seed=0)
    >>> children = [
    ...     Uniform(dummy_child_sampler(1), lower_bound=0, upper_bound=2),
    ...     Uniform(dummy_child_sampler(1), lower_bound=10, upper_bound=12),
    ... ]
    >>> pm = ProductMeasure(sampler=DigitalNetB2(2, seed=9), children=children)
    >>> x = pm(4)
    >>> x.shape
    (4, 2)
    >>> bool(((0 <= x[:, 0]) & (x[:, 0] <= 2)).all())
    True

    The outer sampler controls replications:

    >>> pm = ProductMeasure(
    ...     sampler=DigitalNetB2(2, seed=9, replications=3),
    ...     children=children,
    ... )
    >>> pm(4).shape
    (3, 4, 2)

    The dummy child samplers are only construction placeholders required by
    the current ``AbstractTrueMeasure`` interface. ``ProductMeasure`` samples
    from its own outer sampler.

    Children may have different dimensions:

    >>> import numpy as np
    >>> from qmcpy.true_measure import Gaussian
    >>> children = [
    ...     Gaussian(
    ...         dummy_child_sampler(2),
    ...         mean=[0, 0],
    ...         covariance=np.eye(2),
    ...     ),
    ...     Uniform(dummy_child_sampler(1), lower_bound=10, upper_bound=12),
    ... ]
    >>> pm = ProductMeasure(sampler=DigitalNetB2(3, seed=12), children=children)
    >>> pm(4).shape
    (4, 3)
    """

    def __init__(self, sampler, children):
        """
        Initialize a product measure from one sampler and several children.

        Parameters
        ----------
        sampler : AbstractDiscreteDistribution
            The sampler for the whole product measure. Its dimension must
            equal the sum of the child dimensions.

        children : list or tuple of AbstractTrueMeasure
            Independent true measures to place side by side.

        Why one sampler?
        ----------------
        The product measure should be driven by one total-dimensional QMC
        point set. We do not generate separate QMC samples from each child.
        Instead, one sample u in [0,1]^d is split into blocks:

            u = (u_child_1, u_child_2, ..., u_child_k).

        This preserves the intended total-dimensional QMC construction.
        """
        # A product measure needs at least one child measure.
        if not isinstance(children, (list, tuple)) or len(children) == 0:
            raise ParameterError("ProductMeasure requires a nonempty list of children.")

        # Every child must be a QMCPy true measure so that it has dimension,
        # range, transform, weight, and spawn behavior.
        if not all(isinstance(child, AbstractTrueMeasure) for child in children):
            raise ParameterError(
                "Each ProductMeasure child must be an AbstractTrueMeasure instance."
            )

        # ProductMeasure itself is driven by one sampler for the full product
        # dimension, so the sampler must be a valid QMCPy discrete distribution.
        if not isinstance(sampler, AbstractDiscreteDistribution):
            raise ParameterError(
                "ProductMeasure sampler must be an AbstractDiscreteDistribution."
            )

        self.parameters = ["children"]
        # ProductMeasure uses only the sampler passed directly to ProductMeasure
        # to generate product samples.

        # The child true measures also contain samplers because QMCPy's current
        # AbstractTrueMeasure interface requires true measures to be constructed
        # with an attached discrete distribution. Inside ProductMeasure, those
        # child samplers are not sampled. The children are used for their
        # dimension, range, transform, and weight behavior.

        # A future design could allow samplerless/template child true measures, but
        # that would require a broader AbstractTrueMeasure API decision
        self.children = list(children)

        # Store each child dimension. These dimensions determine where the
        # full sample should be split.
        self.child_dimensions = np.array([child.d for child in self.children], dtype=int)

        # Example: child dimensions [2, 1, 3] give split indices [2, 3].
        # A sample with last axis length 6 is split as [:2], [2:3], [3:].
        self._split_indices = np.cumsum(self.child_dimensions)[:-1]

        # Total product dimension must equal the sampler dimension.
        self._total_child_dimension = int(self.child_dimensions.sum())
        if sampler.d != self._total_child_dimension:
            raise DimensionError(
                "ProductMeasure sampler dimension must equal the sum of child "
                f"dimensions ({sampler.d} != {self._total_child_dimension})."
            )

        # The product measure always starts from the unit cube. A single row
        # [0, 1] is QMCPy's compact way of saying each input coordinate starts
        # in [0, 1].
        self.domain = np.array([[0.0, 1.0]])

        # Let AbstractTrueMeasure parse and store the sampler. This sets
        # attributes such as self.discrete_distrib and self.d.
        self._parse_sampler(sampler)

        # The output range is the stacked range of the children. If a child
        # gives one shared bound row, expand it to one row per child dimension.
        self.range = np.vstack(
            [
                self._expand_bounds(child.range, child.d, "range")
                for child in self.children
            ]
        )

        super(ProductMeasure, self).__init__()

    @staticmethod
    def _expand_bounds(bounds, dimension, name):
        """
        Expand a child's bounds so they have one row per output coordinate.

        Some true measures store bounds as shape (1, 2), meaning the same
        bound applies to all coordinates. Others store bounds as shape
        (dimension, 2), meaning each coordinate has its own bound.

        ProductMeasure needs all child ranges stacked together, so every child
        range must be represented as shape (dimension, 2).
        """
        bounds = np.asarray(bounds)

        # Case 1: one shared bound row, such as [[0, 1]] or [[-inf, inf]].
        # Repeat it once per child coordinate.
        if bounds.shape == (1, 2):
            return np.tile(bounds, (dimension, 1))

        # Case 2: already has one bound row for each child coordinate.
        if bounds.shape == (dimension, 2):
            return bounds

        # Any other shape is ambiguous and would make the final product range
        # incorrect.
        raise DimensionError(
            f"Child true measure {name} must have shape (1, 2) or ({dimension}, 2)."
        )

    @property
    def _has_recursive_child(self):
        """
        Check whether any child is itself recursively composed.

        In QMCPy, a true measure can sometimes be built on top of another true
        measure. Sampling can still be handled by the recursive transform
        helper, but exact product weights in the final transformed space are
        more delicate. For now, ProductMeasure only computes exact weights
        when all children are direct true measures.
        """
        return any(child.transform != child for child in self.children)

    def _split_blocks(self, x):
        """
        Split an input array into child coordinate blocks.

        The split always happens along the final axis, so this works for both

            (n, d)

        and replicated samples such as

            (r, n, d).

        The final axis must equal the total product dimension.
        """
        x = np.asarray(x, dtype=float)

        # The last axis is the coordinate dimension. If it does not match the
        # product dimension, then we cannot assign coordinates to children.
        if x.shape[-1] != self.d:
            raise DimensionError(
                f"ProductMeasure expected last axis {self.d}, got {x.shape[-1]}."
            )

        # Split using the cumulative child dimensions.
        return np.split(x, self._split_indices, axis=-1)

    def _transform(self, x):
        """
        Transform unit-cube samples into product-measure samples.

        Steps
        -----
        1. Split the full unit-cube sample into child blocks.
        2. Send each block to the matching child true measure.
        3. Concatenate the transformed child outputs.

        This implements

            T(u) = (T_1(u_1), T_2(u_2), ..., T_k(u_k)),

        where each child T_j acts only on its own coordinate block.
        """
        blocks = self._split_blocks(x)

        # Use QMCPy's recursive transform helper instead of directly calling
        # child._transform. This keeps sampling compatible with child measures
        # that may already be composed internally.
        transformed_blocks = [
            child._jacobian_transform_r(block, return_weights=False)
            for child, block in zip(self.children, blocks)
        ]

        # Recombine the transformed blocks into one product sample.
        return np.concatenate(transformed_blocks, axis=-1)

    def _weight(self, x):
        """
        Compute the product density/weight for independent child measures.

        For independent components, the joint weight is the product of the
        child weights:

            w(x) = w_1(x_1) * w_2(x_2) * ... * w_k(x_k).

        This method supports direct child true measures. Recursive children
        are blocked for now because their final-space weights need more careful
        handling.
        """
        if self._has_recursive_child:
            raise ParameterError(
                "ProductMeasure exact weights are currently supported only for "
                "direct child true measures."
            )

        blocks = self._split_blocks(x)

        # Weight shape should match all sample axes except the coordinate axis.
        # For x shape (n, d), weight has shape (n,).
        # For x shape (r, n, d), weight has shape (r, n).
        weight = np.ones(np.asarray(x).shape[:-1], dtype=float)

        # Independence means multiply each child weight block by block.
        for child, block in zip(self.children, blocks):
            weight *= child._weight(block)

        return weight

    def _spawn(self, sampler, dimension):
        """
        Spawn a new ProductMeasure with a new sampler.

        QMCPy's spawn mechanism creates new randomized copies of a sampler or
        true measure. ProductMeasure preserves the same child structure and
        replaces only the outer product sampler.

        The dimension cannot change during spawning because changing it would
        invalidate the stored child block structure.
        """
        if dimension != self.d:
            raise DimensionError(
                "ProductMeasure spawning currently preserves the child dimensions."
            )

        return ProductMeasure(sampler=sampler, children=self.children)
