"""
Transforming Low-Discrepancy Sequences from a Cube to a Simplex

This module implements various transformations for converting points from a unit hypercube to a simplex, as described in:

Pillards, T., & Cools, R. (2005). Transforming low-discrepancy sequences from a cube to a simplex. Journal of Computational and Applied Mathematics, 174, 29-42.

Authors: Larysa Matiukha and Sou-Cheng T. Choi
Date: February 6, 2026
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
            >>> len(result)  # Only 1 point satisfies x1 <= x2
            1
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
            >>> len(result)  # All points are preserved
            2
        """
        if points.ndim == 1:
            points = points.reshape(1, -1)
            
        return np.sort(points, axis=1)

