from .._discrete_distribution import LD
from ...util import ParameterError, ParameterWarning
import qmctoolscl
from os.path import dirname, abspath, isfile
from numpy import *
from numpy.lib.npyio import DataSource
import warnings


class DigitalNetB2(LD):
    """
    Quasi-Random digital nets in base 2.
    
    >>> dnb2 = DigitalNetB2(2,seed=7)
    >>> dnb2.gen_samples(4)
    array([[0.05925164, 0.3876318 ],
           [0.9618825 , 0.57022377],
           [0.26831357, 0.90240903],
           [0.74082576, 0.08504858]])
    >>> dnb2.gen_samples(1)
    array([[0.05925164, 0.3876318 ]])
    >>> dnb2
    DigitalNetB2 (DiscreteDistribution Object)
        d               2^(1)
        dvec            [0 1]
        randomize       LMS_DS
        graycode        0
        entropy         7
        spawn_key       ()
    >>> DigitalNetB2(dimension=2,randomize=False,graycode=True).gen_samples(n_min=2,n_max=4,warn=False)
    array([[0.75, 0.25],
           [0.25, 0.75]])
    >>> DigitalNetB2(dimension=2,randomize=False,graycode=False).gen_samples(n_min=2,n_max=4,warn=False)
    array([[0.25, 0.75],
           [0.75, 0.25]])
    >>> dnb2_alpha2 = DigitalNetB2(5,randomize=False,generating_matrices='sobol_mat_alpha2.10600.64.32.lsb.npy')
    >>> dnb2_alpha2.gen_samples(8,warn=False)
    array([[0.      , 0.      , 0.      , 0.      , 0.      ],
           [0.75    , 0.75    , 0.75    , 0.75    , 0.75    ],
           [0.4375  , 0.9375  , 0.1875  , 0.6875  , 0.1875  ],
           [0.6875  , 0.1875  , 0.9375  , 0.4375  , 0.9375  ],
           [0.296875, 0.171875, 0.109375, 0.796875, 0.859375],
           [0.546875, 0.921875, 0.859375, 0.046875, 0.109375],
           [0.234375, 0.859375, 0.171875, 0.484375, 0.921875],
           [0.984375, 0.109375, 0.921875, 0.734375, 0.171875]])
    >>> DigitalNetB2(dimension=3,randomize=False,generating_matrices="LDData/main/dnet/mps.nx_s5_alpha2_m32.txt").gen_samples(8,warn=False)
    array([[0.        , 0.        , 0.        ],
           [0.75841841, 0.45284834, 0.48844557],
           [0.57679828, 0.13226272, 0.10061957],
           [0.31858402, 0.32113875, 0.39369111],
           [0.90278927, 0.45867532, 0.01803333],
           [0.14542431, 0.02548793, 0.4749614 ],
           [0.45587539, 0.33081476, 0.11474426],
           [0.71318879, 0.15377192, 0.37629925]])
    >>> DigitalNetB2(dimension=3,randomize='LMS_DS',seed=5).gen_samples(8)
    array([[0.80884381, 0.15286502, 0.71289026],
           [0.23299976, 0.90131362, 0.02480977],
           [0.6852044 , 0.60716653, 0.36793491],
           [0.35882868, 0.35524087, 0.93003728],
           [0.99508499, 0.76493313, 0.82707006],
           [0.03919028, 0.00898723, 0.38911117],
           [0.6201791 , 0.49853169, 0.23422048],
           [0.4157974 , 0.7422114 , 0.54620031]])
    >>> DigitalNetB2(dimension=3,randomize='DS',seed=5).gen_samples(8)
    array([[0.80500292, 0.80794079, 0.51532556],
           [0.30500292, 0.30794079, 0.01532556],
           [0.55500292, 0.05794079, 0.26532556],
           [0.05500292, 0.55794079, 0.76532556],
           [0.93000292, 0.43294079, 0.89032556],
           [0.43000292, 0.93294079, 0.39032556],
           [0.68000292, 0.68294079, 0.14032556],
           [0.18000292, 0.18294079, 0.64032556]])
    >>> DigitalNetB2(dimension=3,randomize='OWEN',seed=5).gen_samples(8)
    array([[0.33595486, 0.05834975, 0.30066401],
           [0.89110875, 0.84905188, 0.81833285],
           [0.06846074, 0.59997956, 0.67064205],
           [0.6693703 , 0.25824002, 0.10469644],
           [0.44586618, 0.99161977, 0.1873488 ],
           [0.84245267, 0.16445553, 0.56544372],
           [0.18546359, 0.44859876, 0.97389524],
           [0.61215442, 0.64341386, 0.44529863]])
    >>> DigitalNetB2(dimension=3,randomize='LMS',seed=5).gen_samples(8,warn=False)
    array([[0.        , 0.        , 0.        ],
           [0.95589805, 0.75632219, 0.68808121],
           [0.37686047, 0.73555646, 0.9074556 ],
           [0.58124265, 0.49150427, 0.34535346],
           [0.19405368, 0.89429485, 0.39753943],
           [0.76990107, 0.14622242, 0.83549809],
           [0.31964767, 0.346648  , 0.55288926],
           [0.64602387, 0.59820131, 0.2409087 ]])
    >>> DigitalNetB2(dimension=3,randomize='LMS_DS',seed=5,alpha=2).gen_samples(8)
    array([[0.04299256, 0.52996333, 0.47717448],
           [0.94259743, 0.37490772, 0.66679542],
           [0.43054941, 0.40547449, 0.27633971],
           [0.57995435, 0.74852679, 0.58650509],
           [0.33824252, 0.63033195, 0.38382787],
           [0.67517953, 0.4728881 , 0.69863192],
           [0.20188881, 0.25482078, 0.3704321 ],
           [0.78859539, 0.59926902, 0.55590265]])
    >>> DigitalNetB2(dimension=3,randomize='LMS',seed=5,alpha=2).gen_samples(8,warn=False)
    array([[0.        , 0.        , 0.        ],
           [0.97777743, 0.84500686, 0.8148652 ],
           [0.39541554, 0.87656403, 0.23666622],
           [0.62290078, 0.21955672, 0.92207585],
           [0.36557772, 0.15220275, 0.09537604],
           [0.65565677, 0.99469929, 0.78493606],
           [0.22142719, 0.77563873, 0.14448119],
           [0.76124457, 0.12014943, 0.95470772]])
    >>> DigitalNetB2(dimension=3,randomize='False',seed=5,alpha=2).gen_samples(8,warn=False)
    array([[0.      , 0.      , 0.      ],
           [0.75    , 0.75    , 0.75    ],
           [0.4375  , 0.9375  , 0.1875  ],
           [0.6875  , 0.1875  , 0.9375  ],
           [0.296875, 0.171875, 0.109375],
           [0.546875, 0.921875, 0.859375],
           [0.234375, 0.859375, 0.171875],
           [0.984375, 0.109375, 0.921875]])
    >>> DigitalNetB2(dimension=3,randomize='NUS',seed=5,alpha=2).gen_samples(8)
    array([[0.13470966, 0.19119158, 0.96130884],
           [0.97413754, 0.90343374, 0.02884907],
           [0.26275262, 0.86277465, 0.75808903],
           [0.59591889, 0.0264989 , 0.21845493],
           [0.49745711, 0.1158892 , 0.92224289],
           [0.65012418, 0.75808168, 0.09617135],
           [0.11594365, 0.98245561, 0.850095  ],
           [0.7761354 , 0.16906349, 0.15959343]])
    >>> DigitalNetB2(dimension=3,randomize='LMS_DS',seed=7,replications=2).gen_samples(4)
    array([[[0.33268146, 0.39827556, 0.20291175],
            [0.63496292, 0.65168043, 0.98964389],
            [0.14786639, 0.91748989, 0.62849095],
            [0.822218  , 0.1631085 , 0.43090078]],
    <BLANKLINE>
           [[0.05070406, 0.21290819, 0.9154644 ],
            [0.87139488, 0.95651643, 0.11127107],
            [0.45302932, 0.60797654, 0.39932557],
            [0.62525881, 0.35148989, 0.56295289]]])
    >>> DigitalNetB2(dimension=3,randomize='LMS',seed=7,replications=2).gen_samples(4,warn=False)
    array([[[0.        , 0.        , 0.        ],
            [0.96738355, 0.76223591, 0.80730122],
            [0.441163  , 0.55892927, 0.57456673],
            [0.52866341, 0.29766997, 0.3661731 ]],
    <BLANKLINE>
           [[0.        , 0.        , 0.        ],
            [0.82777863, 0.75926807, 0.96154196],
            [0.49614395, 0.67635342, 0.5483692 ],
            [0.67544596, 0.43545693, 0.47763739]]])
    >>> DigitalNetB2(dimension=3,randomize='DS',seed=7,replications=2).gen_samples(4)
    array([[[0.62509547, 0.8972138 , 0.77568569],
            [0.12509547, 0.3972138 , 0.27568569],
            [0.87509547, 0.1472138 , 0.02568569],
            [0.37509547, 0.6472138 , 0.52568569]],
    <BLANKLINE>
           [[0.22520719, 0.30016628, 0.87355345],
            [0.72520719, 0.80016628, 0.37355345],
            [0.47520719, 0.55016628, 0.12355345],
            [0.97520719, 0.05016628, 0.62355345]]])
    >>> DigitalNetB2(dimension=3,randomize='OWEN',seed=7,replications=2).gen_samples(4)
    array([[[0.35967208, 0.89954733, 0.20817919],
            [0.98911419, 0.49964829, 0.62480864],
            [0.02600317, 0.16945373, 0.97946126],
            [0.56008744, 0.54552841, 0.30711945]],
    <BLANKLINE>
           [[0.00782812, 0.71181447, 0.44244982],
            [0.53880628, 0.41910168, 0.65404289],
            [0.29051333, 0.16752862, 0.80878263],
            [0.77449486, 0.83546482, 0.22950922]]])
    
    References:

        [1] Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

        [2] Faure, Henri, and Christiane Lemieux. 
        “Implementation of Irreducible Sobol' Sequences in Prime Power Bases.” 
        Mathematics and Computers in Simulation 161 (2019): 13-22. Crossref. Web.

        [3] F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
        - a survey of analysis and implementation, Foundations of Computational Mathematics, 
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613
        
        [4] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/
        https://bitbucket.org/dnuyens/qmc-generators/src/cb0f2fb10fa9c9f2665e41419097781b611daa1e/cpp/digitalseq_b2g.hpp

        [5] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. 
        (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
        In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), 
        Advances in Neural Information Processing Systems 32 (pp. 8024-8035). Curran Associates, Inc. 
        Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

        [6] I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: 
        "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, 
        Russian Academy of Sciences, Moscow (1992).

        [7] Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 

        [8] Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88-100. 
        DOI:https://doi.org/10.1145/42288.214372
    """

    def __init__(self, dimension=1, randomize='LMS_DS', graycode=False, seed=None, 
        generating_matrices='sobol_mat.21201.32.32.msb.npy', d_max=None, t_max=None, m_max=None, msb=None, t_lms=53, alpha=1,
        _verbose=False, replications=1, qmctoolscl_kwargs={"backend":"c"}):
        r"""
        Args:
            dimension (int or :class:`numpy.ndarray`): dimension of the generator. 
                
                - If an int is passed in, use sequence dimensions [0,...,dimension-1].
                - If an ndarry is passed in, use these dimension indices in the sequence. 

            randomize (bool): apply randomization? True defaults to LMS_DS. Can also explicitly pass in
                
                - "LMS_DS": linear matrix scramble with digital shift
                - "LMS": linear matrix scramble only
                - "DS": digital shift only
                - "OWEN" or "NUS": nested uniform scrambling (Owen scrambling)

            graycode (bool): indicator to use graycode ordering (True) or natural ordering (False)
            seed (list): int seed of list of seeds, one for each dimension.
            generating_matrices (:class:`numpy.ndarray` or str): Specify generating matrices. There are a number of optional input types. 
                
                - A string with either a relative path from `LDData repository <https://github.com/QMCSoftware/LDData>`__ (e.g., "dnet/mps.nx_s5_alpha3_m32.txt") or a NumPy file with format "name.d_max.t_max.m_max.{msb,lsb}.npy" (e.g., "gen_mat.21201.32.32.msb.npy")
                - An ndarray of integers with shape (`d_max`, `m_max)` where each int has `t_max` bits.

            d_max (int): max dimension
            t_max (int): number of bits in each int of each generating matrix, aka: number of rows in a generating matrix with ints expanded into columns
            m_max (int): :math:`2^{\\texttt{m\_max}}` is the number of samples supported, aka: number of columns in a generating matrix with ints expanded into columns
            msb (bool): bit storage as ints, e.g., if :math:`{\\texttt{t\_max}} = 3`, then 6 is [1 1 0] in MSB (True) and [0 1 1] in LSB (False)
            alpha (int): interlacing factor for higher order nets. 
                WARNING: When alpha>1, interlacing is performed regardless of the generating matrices 
                i.e. for alpha>1 do NOT pass in generating matrices which are already interlaced
            t_lms (int): LMS scrambling matrix will be of shape (`t_lms`, `t_max`). Other generating matrix will be of shape (`t_max`, `m_max`).
            _verbose (bool): print randomization details
            replications (int): number of IID randomizations of a pointset
            qmctoolscl_kwargs (dict): keyword arguments for QMCToolsCL to use OpenCL. Defaults to C backend. See https://qmcsoftware.github.io/QMCToolsCL/
        """
        self.parameters = ['dvec','randomize','graycode']
        if alpha>1:
            self.parameters += ["alpha"]
        self.mimics = 'StdUniform'
        self.low_discrepancy = True                      
        self.graycode = graycode
        self.order = "GRAY" if self.graycode else "NATURAL"
        self.replications_gm = 1
        # generating matrices 
        if isinstance(generating_matrices,ndarray):
            if generating_matrices.ndim==2:
                self.z_og = generating_matrices[None,:,:]
            else:
                assert generating_matrices.ndim==3
                self.z_og = generating_matrices
                self.replications_gm = len(self.z_og)
            if d_max is None or t_max is None or m_max is None or msb is None:
                raise ParameterError("d_max, t_max, m_max, and msb must be supplied when generating_matrices is a ndarray")
            self.d_max = d_max
            self.t_max = t_max
            self.m_max = m_max
            self.msb = msb
        elif isinstance(generating_matrices,str):
            parts = generating_matrices.split('.')
            root = dirname(abspath(__file__))+'/generating_matrices/'
            repos = DataSource()
            if generating_matrices[-4:]==".txt":
                if repos.exists(generating_matrices):
                    datafile = repos.open(generating_matrices)
                elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices):
                    datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices)
                elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"+generating_matrices):
                    datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"+generating_matrices)
                elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices[7:]):
                    datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices[7:])
                elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/"+generating_matrices):
                    datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/"+generating_matrices)
                else:
                    raise ParameterError("LDData path %s not found"%generating_matrices)
                contents = [line.rstrip('\n').strip() for line in datafile.readlines()]
                self.msb = True
                contents = [line.split("#",1)[0] for line in contents if line[0]!="#"]
                datafile.close()
                assert int(contents[0])==2, "DigitalNetB2 requires base=2 " # base 2
                self.d_max = int(contents[1])
                self.m_max = int(log2(int(contents[2])))
                self.t_max = int(contents[3])
                compat_shift = self.t_max-64 if self.t_max>=64 else 0
                if compat_shift>0: warnings.warn("Truncating ints in generating matrix to have 64 bits.")
                self.z_og = array([[int(v)>>compat_shift for v in line.split(' ')] for line in contents[4:]],dtype=uint64)[None,:]
            elif isfile(root+generating_matrices):
                self.z_og = load(root+generating_matrices).astype(uint64)[None,:]
                self.d_max = int(parts[-5])
                self.t_max = int(parts[-4])
                self.m_max = int(parts[-3])
                self.msb = bool(parts[-2].lower()=='msb')
            elif isfile(generating_matrices):
                self.z_og = load(generating_matrices).astype(uint64)[None,:]
                self.d_max = int(parts[-5])
                self.t_max = int(parts[-4])
                self.m_max = int(parts[-3])
                self.msb = bool(parts[-2].lower()=='msb')
            else:
                raise ParameterError("generating_matrices '%s' not found."%generating_matrices)
        else:
            raise ParameterError("invalid input for generating_matrices, see documentation and doctests.")
        super(DigitalNetB2,self).__init__(dimension,seed)
        if not self.msb:
            qmctoolscl.dnb2_gmat_lsb_to_msb(uint64(self.replications_gm),uint64(self.d),uint64(self.m_max),tile(uint64(self.t_max),self.replications_gm),self.z_og,self.z_og,**qmctoolscl_kwargs)
        # randomizations
        self.t_lms = self.t_max if self.t_max>t_lms else t_lms
        assert self.t_max<=64 and self.t_lms<=64, "require t_max <= t_lms <= 64"
        self.alpha = alpha
        self._verbose = _verbose
        self.randomize = str(randomize).upper()
        if self.randomize=="TRUE": self.randomize = "LMS_DS"
        if self.randomize=="OWEN": self.randomize = "NUS"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["LMS_DS","LMS","DS","NUS","FALSE"]
        self.z = self.z_og[:,self.dvec].copy()
        if self.alpha>1 and self.randomize in ["DS","FALSE"]:
            assert (self.dvec==arange(self.d)).all()
            dtalpha = self.alpha*self.d
            z_ho = empty((self.replications_gm,self.d,self.m_max),dtype=uint64)
            qmctoolscl.dnb2_interlace(uint64(self.replications_gm),uint64(self.d),uint64(self.m_max),uint64(dtalpha),uint64(self.t_max),uint64(self.t_lms),uint64(self.alpha),self.z_og[:,:dtalpha].copy(),z_ho,**qmctoolscl_kwargs)
            self.z = z_ho
        if "LMS" in self.randomize:
            if self.alpha==1:
                z_lms = empty((replications,self.d,self.m_max),dtype=uint64)
                S = qmctoolscl.dnb2_get_linear_scramble_matrix(self.rng,uint64(replications),uint64(self.d),uint64(self.t_max),uint64(self.t_lms),uint64(self._verbose))
                qmctoolscl.dnb2_linear_matrix_scramble(uint64(replications),uint64(self.d),uint64(self.m_max),uint64(self.replications_gm),uint64(self.t_lms),S,self.z,z_lms,**qmctoolscl_kwargs)
                self.z = z_lms 
                self.replications_gm = replications
            else:
                assert (self.dvec==arange(self.d)).all()
                dtalpha = self.alpha*self.d
                z_lms = empty((replications,dtalpha,self.m_max),dtype=uint64)
                S = qmctoolscl.dnb2_get_linear_scramble_matrix(self.rng,uint64(replications),uint64(dtalpha),uint64(self.t_max),uint64(self.t_lms),uint64(self._verbose))
                qmctoolscl.dnb2_linear_matrix_scramble(uint64(replications),uint64(dtalpha),uint64(self.m_max),uint64(self.replications_gm),uint64(self.t_lms),S,self.z_og[:,:dtalpha].copy(),z_lms,**qmctoolscl_kwargs)
                z_lms_ho = empty((replications,self.d,self.m_max),dtype=uint64)
                qmctoolscl.dnb2_interlace(uint64(replications),uint64(self.d),uint64(self.m_max),uint64(dtalpha),uint64(self.t_lms),uint64(self.t_lms),uint64(self.alpha),z_lms,z_lms_ho,**qmctoolscl_kwargs)
                self.z = z_lms_ho
                self.replications_gm = replications
        if "DS" in self.randomize:
            self.rshift = qmctoolscl.random_tbit_uint64s(self.rng,self.t_lms,(replications,self.d))
        if "NUS" in self.randomize:
            if alpha==1:
                new_seeds = self._base_seed.spawn(replications*self.d)
                self.rngs = array([random.Generator(random.SFC64(new_seeds[j])) for j in range(replications*self.d)]).reshape(replications,self.d)
                self.root_nodes = array([qmctoolscl.NUSNode_dnb2() for i in range(replications*self.d)]).reshape(replications,self.d)
            else:
                assert (self.dvec==arange(self.d)).all()
                self.dtalpha = self.alpha*self.d
                new_seeds = self._base_seed.spawn(replications*self.dtalpha)
                self.rngs = array([random.Generator(random.SFC64(new_seeds[j])) for j in range(replications*self.dtalpha)]).reshape(replications,self.dtalpha)
                self.root_nodes = array([qmctoolscl.NUSNode_dnb2() for i in range(replications*self.dtalpha)]).reshape(replications,self.dtalpha)
                self.z = self.z_og[:,:self.dtalpha].copy()
        if self.replications_gm>1: assert replications==self.replications_gm, "if replications_gm>1 require replications = replications_gm"
        self.replications = replications

    def gen_samples(self, n=None, n_min=0, n_max=8, warn=True, return_unrandomized=False, return_binary=False, qmctoolscl_gen_kwargs={"backend":"c"}, qmctoolscl_rand_kwargs={"backend":"c"}, qmctoolscl_convert_kwargs={"backend":"c"}):
        """
        Generate samples

        Args:
            n (int): if n is supplied, generate from n_min=0 to n_max=n samples. 
                Otherwise use the n_min and n_max explicitly supplied as the following 2 arguments
            n_min (int): Starting index of sequence.
            n_max (int): Final index of sequence.
            return_unrandomized (bool): return randomized samples, unrandomized samples or just randomized samples? 
            return_binary (bool): return binary samples as well? 
            qmctoolscl_gen_kwargs,qmctoolscl_rand_kwargs,qmctoolscl_convert_kwargs (dict): keyword arguments for QMCToolsCL to use OpenCL when generating points, performing randomizations, and converting to floats. Defaults to C backend. See https://qmcsoftware.github.io/QMCToolsCL/


        Returns:
            ndarray: replications x (n_max-n_min) x d (dimension) array of samples
        """
        if n:
            n_min = 0
            n_max = n
        if n_min == 0 and warn and self.randomize in ["FALSE","LMS"]:
            warnings.warn("Non-randomized DigitalNetB2 sequence includes the origin",ParameterWarning)
        if n_max > 2**self.m_max:
            raise ParameterError('DigitalNetB2 generating matrices support up to %d samples.'%(2**self.m_max))
        r_x = uint64(self.replications_gm) 
        n = uint64(n_max-n_min)
        d = uint64(self.dtalpha) if self.randomize=="NUS" and self.alpha>1 else uint64(self.d)
        n_start = uint64(n_min)
        mmax = uint64(self.m_max)
        xb = empty((r_x,n,d),dtype=uint64)
        if self.graycode:
            _ = qmctoolscl.dnb2_gen_gray(r_x,n,d,n_start,mmax,self.z,xb,**qmctoolscl_gen_kwargs)
        else: 
            assert (n_min==0 or log2(n_min)%1==0) and (n_max==0 or log2(n_max)%1==0), "DigitalNetB2 in natural order requires n_min and n_max be 0 or powers of 2"
            _ = qmctoolscl.dnb2_gen_natural(r_x,n,d,n_start,mmax,self.z,xb,**qmctoolscl_gen_kwargs)
        if return_unrandomized or self.randomize in ["FALSE","LMS"]:
            tmaxes_new = tile(uint64(self.t_max if self.randomize=="FALSE" and self.alpha==1 else self.t_lms),int(r_x))
            if not return_binary:
                x = empty((r_x,n,d),dtype=float64)
                _ = qmctoolscl.dnb2_integer_to_float(r_x,n,d,tmaxes_new,xb,x,**qmctoolscl_convert_kwargs)
            if r_x==1: 
                if not return_binary:
                    x = x[0]
                else:
                    xb = xb[0]
        r = uint64(self.replications)
        if "NUS" in self.randomize:
            assert return_unrandomized==False
            if self.alpha==1:
                xrb = empty((r,n,d),dtype=uint64)
                tmax = uint64(self.t_max)
                tmax_new = uint64(self.t_lms)
                _ = qmctoolscl.dnb2_nested_uniform_scramble(r,n,d,r_x,tmax,tmax_new,self.rngs,self.root_nodes,xb,xrb)
                if not return_binary:
                    tmaxes_new = tile(uint64(self.t_lms),int(r))
                    xr = empty((r,n,d),dtype=float64)
                    _ = qmctoolscl.dnb2_integer_to_float(r,n,d,tmaxes_new,xrb,xr,**qmctoolscl_convert_kwargs)
            else:
                d = uint64(self.d)
                dtalpha = uint64(self.dtalpha)
                alpha = uint64(self.alpha)
                xrb_lo = empty((r,n,self.dtalpha),dtype=uint64)
                tmax = uint64(self.t_max)
                tmax_new = uint64(self.t_lms)
                _ = qmctoolscl.dnb2_nested_uniform_scramble(r,n,dtalpha,r_x,tmax,tmax_new,self.rngs,self.root_nodes,xb,xrb_lo)
                xrb_t = empty((r,d,n),dtype=uint64)
                xrb_lo_t = moveaxis(xrb_lo,[1,2],[2,1]).copy() 
                qmctoolscl.dnb2_interlace(r,d,n,dtalpha,tmax_new,tmax_new,alpha,xrb_lo_t,xrb_t)
                xrb = moveaxis(xrb_t,[1,2],[2,1]).copy()
                if not return_binary:
                    tmaxes_new = tile(uint64(self.t_lms),int(r))
                    xr = zeros((r,n,d),dtype=float64)
                    qmctoolscl.dnb2_integer_to_float(r,n,d,tmaxes_new,xrb,xr)
            if r==1:
                if not return_binary:
                    xr=xr[0]
                else:
                    xrb = xrb[0]
        if "DS" in self.randomize:
            xrb = empty((r,n,d),dtype=uint64)
            lshifts = tile(uint64((self.t_lms-self.t_max) if "LMS" not in self.randomize else 0),int(r))
            _ = qmctoolscl.dnb2_digital_shift(r,n,d,r_x,lshifts,xb,self.rshift,xrb,**qmctoolscl_rand_kwargs)
            if not return_binary:
                tmaxes_new = tile(uint64(self.t_lms),int(r))
                xr = empty((r,n,d),dtype=float64)
                _ = qmctoolscl.dnb2_integer_to_float(r,n,d,tmaxes_new,xrb,xr,**qmctoolscl_convert_kwargs)
            if r==1:
                if not return_binary:
                    xr=xr[0]
                else:
                    xrb = xrb[0]
        if self.randomize in ["FALSE","LMS"]:
            if return_binary: 
                return xb
            elif return_unrandomized:
                raise ParameterError("return_unrandomized=True only applies when when randomize='SHIFT'.")
            else:
                return x
        else:
            if return_binary: 
                return xrb
            elif return_unrandomized:
                return xr,x 
            else:
                return xr
    
    def pdf(self, x):
        """ pdf of a standard uniform """
        return ones(x.shape[0], dtype=float) 
            
    def _spawn(self, child_seed, dimension):
        return DigitalNetB2(
            dimension=dimension,
            randomize=self.randomize,
            graycode=self.graycode,
            seed=child_seed,
            generating_matrices=self.z_og,
            d_max=self.d_max,
            t_max=self.t_max,
            m_max=self.m_max,
            msb=True, # self.z_og is put into MSB during first initialization 
            t_lms=self.t_lms,
            replications=self.replications)

class Sobol(DigitalNetB2): pass

