from abc import ABC, abstractmethod
import numpy as np


# Specify and generate values $f(\vx)$ for $\vx \in \cx$
class fun(ABC):

    def __init__(self):
        self.domain = np.array([[0, 0], [1, 1]])  # domain of the function, $\cx$
        self.domainType = 'box'  # e.g., 'box', 'ball'
        self.dimension = 2  # dimension of the domain, $d$
        self.distribType = 'uniform'  # e.g., 'uniform', 'Gaussian'
        self.nominalValue = 0  # a nominal number, $c$, such that $(c, \ldots, c) \in \cx$
        super().__init__()

    @abstractmethod
    def f(self, x, coordIndex):
        '''
         x = nodes, $\vx_{\fu,i} = i^{\text{th}}$ row of an $n \times |\fu|$ matrix
         coordIndex = set of those coordinates in sequence needed, $\fu$
         y = $n \times p$ matrix with values $f(\vx_{\fu,i},\vc)$ where
            if $\vx_i' = (x_{i,\fu},\vc)_j$, then $x'_{ij} = x_{ij}$ for $j \in \fu$, and $x'_{ij} = c$ otherwise
        '''
        pass
