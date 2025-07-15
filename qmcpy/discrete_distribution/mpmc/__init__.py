"""
Message Passing Monte Carlo (MPMC) discrete distribution.

This module implements MPMC using PyTorch and PyTorch Geometric for 
generating low-discrepancy point sets through neural message passing.
"""

from .mpmc import MPMC

__all__ = ['MPMC']
