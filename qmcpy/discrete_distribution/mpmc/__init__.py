"""
Message Passing Monte Carlo (MPMC) discrete distribution.

This module implements MPMC using PyTorch and PyTorch Geometric for 
generating low-discrepancy point sets through neural message passing.
"""

try: 
    import torch
    import torch_cluster
    import torch_geometric
    from .mpmc import MPMC
except ImportError:
    class MPMC(object):
        def __init__(self, *args, **kwargs):
            raise Exception("MPMC requires torch, torch_cluster, and torch_geometric but no installations found")

__all__ = ['MPMC']
