from .._discrete_distribution import DiscreteDistribution
from ...util import ParameterError
from .sobol_qrng import SobolQRNG
from .sobol_pytorch import SobolPyTorch
from .sobol_seq51 import SobolSeq51


class Sobol(DiscreteDistribution):
    """
    Quasi-Random Sobol nets in base 2.
    
    >>> s = Sobol(2,seed=7)
    >>> s
    Sobol (DiscreteDistribution Object)
        dimension       2^(1)
        randomize       1
        graycode        0
        seed            7
        mimics          StdUniform
        backend         QRNG
    >>> s.gen_samples(4)
    array([[0.982, 0.883],
           [0.482, 0.383],
           [0.732, 0.133],
           [0.232, 0.633]])
    >>> s.set_dimension(3)
    >>> s.gen_samples(n_min=4,n_max=8)
    array([[0.857, 0.258, 0.226],
           [0.357, 0.758, 0.726],
           [0.607, 0.508, 0.976],
           [0.107, 0.008, 0.476]])
    >>> Sobol(dimension=2,randomize=False,graycode=True).gen_samples(n_min=2,n_max=4)
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> Sobol(dimension=2,randomize=False,graycode=False).gen_samples(n_min=2,n_max=4)
    array([[0.25, 0.75],
           [0.75, 0.25]])
           
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

        [2] Faure, Henri, and Christiane Lemieux. 
        “Implementation of Irreducible Sobol' Sequences in Prime Power Bases.” 
        Mathematics and Computers in Simulation 161 (2019): 13–22. Crossref. Web.

        [3] F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
        - a survey of analysis and implementation, Foundations of Computational Mathematics, 
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613
        
        [4] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/

        [5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. 
        (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
        In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), 
        Advances in Neural Information Processing Systems 32 (pp. 8024–8035). Curran Associates, Inc. 
        Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

        [6] I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: 
        "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, 
        Russian Acamdey of Sciences, Moscow (1992).

        [7] Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 
    """
    
    parameters = ['dimension','randomize','graycode','seed','mimics','backend']

    def __init__(self, dimension=1, randomize=True, graycode=False, seed=None, backend='QRNG'):
        """
        Args:
            dimension (int): dimension of samples
            randomize (bool): If True, apply digital shift to generated samples.
                Note: Non-randomized Sobol' sequence includes the origin.
            seed (int): seed the random number generator for reproducibility
            backend (str): backend generator must be either "QRNG", "MPS", "PyTorch", or "Seq51"
                "QRNG" is significantly faster, supports optional randomization, and supports optional graycode ordering.
            graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
        """
        self.backend = backend.upper()
        backend_objs = {'QRNG':SobolQRNG, 'PYTORCH':SobolPyTorch, 'SEQ51':SobolSeq51}
        backends = list(backend_objs.keys())
        if self.backend not in backends:
            raise ParameterError('Sobol requires backend be in %s'%(str(backends)))
        self.generator = backend_objs[self.backend](dimension,randomize,graycode,seed)
        self.dimension, self.randomize, self.graycode, self.seed = self.generator.get_params()
        self.low_discrepancy = True
        self.mimics = 'StdUniform'
        super(Sobol,self).__init__()

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.

        Returns:
            ndarray: (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        x = self.generator.gen_samples(n_min,n_max,warn)
        return x

    def set_seed(self, seed):
        """ See abstract method. """
        self.seed = self.generator.set_seed(seed)
        
    def set_dimension(self, dimension):
        """ See abstract method. """
        self.dimension = self.generator.set_dimension(dimension)
