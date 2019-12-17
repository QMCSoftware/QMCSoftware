"""
This package is a refactored version of Dirk Nuyens' Magic Point Shop package
maintained in full at /third_party_magic_point_shop/
Reference:
    F.Y. Kuo & D. Nuyens.
    Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
    - a survey of analysis and implementation, Foundations of Computational Mathematics, 
    16(6):1631-1696, 2016.
    springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
    arxiv link: https://arxiv.org/abs/1606.06613
Online Link: https://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/
"""

# API
from .digital_sequence import DigitalSeq
from .lattice_sequence import LatticeSeq