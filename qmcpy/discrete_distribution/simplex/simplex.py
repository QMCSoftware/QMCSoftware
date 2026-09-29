"""
Transformations of points from a unit hypercube onto a simplex.

Implements Drop, Sort, Root, and Mirror as given in [1], and Origami and Shift
as given in the fuller derivation in Chapter 4 ("Transformations for a
Simplex") of [2].

Authors: Larysa Matiukha and Sou-Cheng T. Choi
Date: February 6, 2026

Unit tests:
    python -W ignore -m unittest test.test_dd_discrete_distribs.TestSimplexTransform -v

References:
    [1] T. Pillards and R. Cools, "Transforming low-discrepancy sequences from a cube
    to a simplex," *Journal of Computational and Applied Mathematics*, vol. 174, no. 1,
    pp. 29-42, 2005.

    [2] T. Pillards, "Quasi-Monte Carlo integration over a simplex and the entire
    space," Ph.D. thesis, KU Leuven, 2006. [Online]. Available:
    https://www.cs.kuleuven.be/publicaties/doctoraten/tw/TW2006_05.pdf
"""

import numpy as np


class SimplexTransform:
    """
    A class implementing various transformations from the unit cube to a simplex.
    
    The simplex Ts is defined as:
    Ts = {(x1, ..., xs) ∈ Rs : 0 ≤ x1 ≤ x2 ≤ ... ≤ xs ≤ 1}
    
    Attributes:
        dimension (int): The dimension of the space
    """

    def __init__(self, dimension: int = 2):
        """
        Initialize the SimplexTransform class.
        
        Args:
            dimension (int): The dimension of the space (default: 2)
        """
        self.dimension = dimension

    def drop(self, points: np.ndarray) -> np.ndarray:
        """
        Transformation Drop: Keep only points that fall inside the simplex.
        
        This is a straightforward but inefficient transformation. Only 1 out of s! 
        points is kept in higher dimensions.
        
        Args:
            points (np.ndarray): Points in the unit cube, shape (N, s)
            
        Returns:
            np.ndarray: Points that fall inside the simplex
            
        Examples:
            >>> import numpy as np
            >>> transformer = SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> result = transformer.drop(points)
            >>> result
            array([[0.3, 0.7]])
        """
        # Check if points satisfy x1 ≤ x2 ≤ ... ≤ xs
        if points.ndim == 1:
            points = points.reshape(1, -1)

        mask = np.all(points[:, :-1] <= points[:, 1:], axis=1)
        return points[mask]

    def sort(self, points: np.ndarray) -> np.ndarray:
        """
        Transformation Sort: Sort the coordinates of each point.
        
        This is a fast, continuous transformation that recovers points lost by Drop.
        When we sort the coordinates of a point in Is (such that xi ≤ xi+1), 
        we obtain a point in the simplex Ts.
        
        Args:
            points (np.ndarray): Points in the unit cube, shape (N, s)
            
        Returns:
            np.ndarray: Transformed points in the simplex
            
        Examples:
            >>> import numpy as np
            >>> transformer = SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> result = transformer.sort(points)
            >>> result
            array([[0.3, 0.7],
                   [0.4, 0.8]])
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)

        return np.sort(points, axis=1)

    def root(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Root: map points via the cumulative distribution function.

        Based on [2], Sec. 4.3.5: a bijective, continuous transformation for
        any dimension d, with no free parameters.
        Writing the input as (x1, ..., xd), the output (y1, ..., yd) is

            yd := xd ** (1/d)
            y_i := y_{i+1} * x_i ** (1/i)   for i = d-1, ..., 1

        Root is highly non-uniform: points near x_d = 0 get shifted far more
        than points near x_d = 1.

        Args:
            points (np.ndarray): Points in the unit cube, shape (N, s)

        Returns:
            np.ndarray: Transformed points in the simplex

        Examples:
            >>> import numpy as np
            >>> transformer = SimplexTransform(dimension=2)
            >>> transformer.root(np.array([0.5, 0.01]))
            array([[0.05, 0.1 ]])
            >>> np.round(transformer.root(np.array([0.5, 0.99])), 3)
            array([[0.497, 0.995]])
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)

        points = np.asarray(points, dtype=float)
        d = points.shape[1]
        y = np.empty_like(points)
        y[:, d - 1] = points[:, d - 1] ** (1.0 / d)
        for i in range(d - 2, -1, -1):
            y[:, i] = y[:, i + 1] * points[:, i] ** (1.0 / (i + 1))
        return y

    def mirror(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Mirror: keep points already in the simplex fixed and
        reflect every other point into it.

        Based on [2], Sec. 4.3.3. Only dimensions 1-3 are given a closed form
        there; Mirror is fast but discontinuous, and (unlike Root or Shift)
        folds half of any point set that is symmetric about its center (e.g. a
        lattice) on top of the other half.

        Args:
            points (np.ndarray): Points in the unit cube, shape (N, s), s in {1, 2, 3}

        Returns:
            np.ndarray: Transformed points in the simplex

        Raises:
            NotImplementedError: if the points have dimension greater than 3

        Examples:
            >>> import numpy as np
            >>> transformer = SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> transformer.mirror(points)
            array([[0.3, 0.7],
                   [0.2, 0.6]])
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)

        y = np.array(points, dtype=float, copy=True)
        d = y.shape[1]
        if d == 1:
            return y
        if d == 2:
            swap = y[:, 0] > y[:, 1]
            y[swap] = 1.0 - y[swap]
            return y
        if d == 3:
            x1, x2, x3 = y[:, 0].copy(), y[:, 1].copy(), y[:, 2].copy()
            m = x3 <= x1
            x1[m], x2[m], x3[m] = 1.0 - x1[m], 1.0 - x2[m], 1.0 - x3[m]
            m = x3 <= x2
            x2[m], x3[m] = 1.0 - x2[m] + x1[m], 1.0 - x3[m] + x1[m]
            m = x2 <= x1
            x1_new, x2_new = x3[m] - x1[m], x3[m] - x2[m]
            x1[m], x2[m] = x1_new, x2_new
            return np.stack([x1, x2, x3], axis=1)
        raise NotImplementedError(
            "Transformation Mirror is only given a closed form for dimension 1-3 in "
            "Pillards & Cools (2005); no general-d formula is given there for d > 3."
        )

    def origami(self, points: np.ndarray, base: int = 2, depth: int = 1) -> np.ndarray:
        r"""
        Transformation Origami: recursively apply Sort within a grid of cubes,
        from the finest scale down to the whole cube.

        Based on [2], Sec. 4.3.4. Choosing a base b and depth m (so the finest
        grid has M = b**m cells per axis), Origami divides the unit cube into
        M**d cells and applies Sort within each; then repeats at grid
        resolutions M/b, M/b**2, ..., b, 1, always operating on the current
        (already partly transformed) point, with the final N=1 pass equal to a
        plain global Sort. Origami is discontinuous but, unlike Sort, keeps
        every elementary interval (see [2], Sec. 2.3.5) the same size after the
        transformation. depth=0 reduces to plain Sort.

        Args:
            points (np.ndarray): Points in the unit cube, shape (N, s)
            base (int): grid subdivisions per level, b >= 2
            depth (int): number of levels above the base grid, m >= 0

        Returns:
            np.ndarray: Transformed points in the simplex

        Examples:
            >>> import numpy as np
            >>> transformer = SimplexTransform(dimension=2)
            >>> transformer.origami(np.array([0.9, 0.3]), base=2, depth=1)
            array([[0.4, 0.8]])
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)

        x = np.array(points, dtype=float, copy=True)
        b = int(base)
        for n in (b ** k for k in range(depth, -1, -1)):
            cell = np.floor(n * x)
            frac = n * x - cell
            frac.sort(axis=1)
            x = (cell + frac) / n
        return x

    def shift(self, points: np.ndarray) -> np.ndarray:
        r"""
        Transformation Shift: push the unit cube into the simplex Ad (Eq. 4.5),
        then map Ad onto the simplex Td used elsewhere in this class (Eq. 4.6).

        Based on [2], Sec. 4.3.6. Sorting the input ascending, then for
        k = 1, ..., d-1 subtracting (d-k)/(d-k+1) times the gap between sorted
        coordinates k-1 and k from every sorted coordinate at or after
        position k, maps the unit cube onto Ad = {x in Id : sum(x) < 1};
        unsorting and taking the cumulative sum (Eq. 4.6) then maps Ad onto
        Td. Shift is continuous for any dimension d, with no free parameters,
        and (per the elementary-interval comparison in [2], Sec. 4.4) keeps
        elementary intervals more compact than Root does.

        Args:
            points (np.ndarray): Points in the unit cube, shape (N, s)

        Returns:
            np.ndarray: Transformed points in the simplex

        Examples:
            >>> import numpy as np
            >>> transformer = SimplexTransform(dimension=2)
            >>> points = np.array([[0.3, 0.7], [0.8, 0.4]])
            >>> transformer.shift(points)
            array([[0.15, 0.7 ],
                   [0.6 , 0.8 ]])
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)

        points = np.asarray(points, dtype=float)
        d = points.shape[1]
        if d == 1:
            return points.copy()

        order = np.argsort(points, axis=1)
        x = np.take_along_axis(points, order, axis=1)
        for j in range(d - 1):
            prev = x[:, j - 1] if j >= 1 else 0.0
            gap = x[:, j] - prev
            coeff = (d - j - 1) / (d - j)
            x[:, j:] -= (coeff * gap)[:, None]
        inverse_order = np.argsort(order, axis=1)
        a = np.take_along_axis(x, inverse_order, axis=1)
        return np.cumsum(a, axis=1)

