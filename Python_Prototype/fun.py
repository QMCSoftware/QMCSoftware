from abc import ABC, abstractmethod
import numpy as np


# Â§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$Â§
class fun(ABC):

    def __init__(self):
        self.domain = np.array([[0, 0], [1, 1]])  # domain of the function, Â§$\mcommentfont \cx$Â§
        self.domainType = 'box'  # e.g., 'box', 'ball'
        self.dimension = 2  # dimension of the domain, Â§$\mcommentfont d$Â§
        self.distribType = 'uniform'  # e.g., 'uniform', 'Gaussian'
        self.nominalValue = 0  # a nominal number, Â§$\mcommentfont c$Â§, such that Â§$\mcommentfont (c, \ldots, c) \in \cx$Â§

    @abstractmethod
    def f(self, x, coordIndex):
        '''
         x = nodes, Â§\mcommentfont $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrixÂ§
         coordIndex = set of those coordinates in sequence needed, Â§\mcommentfont $\fu$Â§
         y = Â§\mcommentfont$n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where
            if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwiseÂ§
        '''
        pass
