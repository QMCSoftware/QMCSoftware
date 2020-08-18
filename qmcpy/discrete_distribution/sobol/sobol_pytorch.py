from ...util import ParameterError, ParameterWarning
from numpy import *
import warnings


class SobolPyTorch(object):
    """
    References:

        [1] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. 
        (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
        In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), 
        Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. 
        Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf
    """

    def __init__(self, dimension, randomize, graycode, seed):
        warnings.warn('''
                PyTorch Sobol issue. See https://github.com/pytorch/pytorch/issues/32047
                    SobolEngine 0^{th} vector is \\vec{.5} rather than \\vec{0}''',ParameterWarning)
        self.r = randomize
        self.g = graycode
        if not self.g:
            raise ParameterError('PyTorch Sobol only supports Graycode ordering. Use "QRNG" backend for natural ordering.')
        self.set_seed(seed)
        self.set_dimension(dimension)

    def gen_samples(self, n_min, n_max, warn):
        import torch
        self.se.reset()
        self.se.fast_forward(n_min)
        n = int(n_max-n_min)
        x = self.se.draw(n,dtype=torch.float64).numpy()
        return x

    def set_seed(self, seed):
        self.s = seed if seed else random.randint(2**32)
        return self.s
        
    def set_dimension(self, dimension):
        import torch
        self.d = dimension
        self.se = torch.quasirandom.SobolEngine(dimension=self.d, scramble=self.r, seed=self.s)
        return self.d

    def get_params(self):
        return self.d, self.r, self.g, self.s