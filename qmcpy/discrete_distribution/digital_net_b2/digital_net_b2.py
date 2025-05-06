from ..abstract_discrete_distribution import AbstractLDDiscreteDistribution
from ...util import ParameterError, ParameterWarning
import qmctoolscl
from os.path import dirname, abspath, isfile
import numpy as np
from numpy.lib.npyio import DataSource
import warnings
from copy import deepcopy


class DigitalNetB2(AbstractLDDiscreteDistribution):
    """
    Low discrepancy digital net in base 2.

    Note:
        `Sobol` is an alias for `DigitalNetB2`
    
    Examples:
        >>> dnb2 = DigitalNetB2(2,seed=7)
        >>> dnb2(4)
        array([[0.84429662, 0.72162356],
               [0.1020178 , 0.08593631],
               [0.6019625 , 0.27339078],
               [0.34430233, 0.90916911]])
        >>> dnb2(1)
        array([[0.84429662, 0.72162356]])
        >>> dnb2
        DigitalNetB2 (DiscreteDistribution Object)
            d               2^(1)
            replications    1
            randomize       LMS_DS
            gen_mats_source joe_kuo.6.21201.txt
            order           NATURAL
            t               63
            alpha           1
            n_limit         2^(32)
            entropy         7
        >>> DigitalNetB2(dimension=2,randomize=False,order="GRAY")(n_min=2,n_max=4,warn=False)
        array([[0.75, 0.25],
               [0.25, 0.75]])
        >>> DigitalNetB2(dimension=2,randomize=False,order="NATURAL")(n_min=2,n_max=4,warn=False)
        array([[0.25, 0.75],
               [0.75, 0.25]])
        >>> DigitalNetB2(dimension=3,randomize=False,generating_matrices="LDData/main/dnet/mps.nx_s5_alpha2_m32.txt")(8,warn=False)
        array([[0.        , 0.        , 0.        ],
               [0.75841841, 0.45284834, 0.48844557],
               [0.57679828, 0.13226272, 0.10061957],
               [0.31858402, 0.32113875, 0.39369111],
               [0.90278927, 0.45867532, 0.01803333],
               [0.14542431, 0.02548793, 0.4749614 ],
               [0.45587539, 0.33081476, 0.11474426],
               [0.71318879, 0.15377192, 0.37629925]])
        >>> DigitalNetB2(dimension=3,randomize='LMS_DS',seed=5)(8)
        array([[4.42642214e-01, 6.86933174e-01, 6.93464013e-01],
               [9.45171550e-01, 4.46137011e-01, 2.00191886e-01],
               [2.48950322e-01, 8.83791037e-02, 3.52266650e-01],
               [7.46535684e-01, 7.94140908e-01, 8.46057454e-01],
               [3.08394915e-01, 3.05848350e-01, 9.45654253e-01],
               [8.05988022e-01, 5.76671664e-01, 4.40177478e-01],
               [3.78447987e-04, 9.65679803e-01, 1.00564692e-01],
               [5.02915291e-01, 1.67390371e-01, 6.05583580e-01]])
        >>> DigitalNetB2(dimension=3,randomize='DS',seed=5)(8)
        array([[0.68383949, 0.04047995, 0.42903182],
               [0.18383949, 0.54047995, 0.92903182],
               [0.93383949, 0.79047995, 0.67903182],
               [0.43383949, 0.29047995, 0.17903182],
               [0.55883949, 0.66547995, 0.05403182],
               [0.05883949, 0.16547995, 0.55403182],
               [0.80883949, 0.41547995, 0.80403182],
               [0.30883949, 0.91547995, 0.30403182]])
        >>> DigitalNetB2(dimension=3,randomize='OWEN',seed=5)(8)
        array([[0.33595486, 0.05834975, 0.30066401],
               [0.89110875, 0.84905188, 0.81833285],
               [0.06846074, 0.59997956, 0.67064205],
               [0.6693703 , 0.25824002, 0.10469644],
               [0.44586618, 0.99161977, 0.1873488 ],
               [0.84245267, 0.16445553, 0.56544372],
               [0.18546359, 0.44859876, 0.97389524],
               [0.61215442, 0.64341386, 0.44529863]])
        >>> DigitalNetB2(dimension=3,randomize='LMS',seed=5)(8,warn=False)
        array([[0.        , 0.        , 0.        ],
               [0.50255985, 0.86689489, 0.51063711],
               [0.30826124, 0.72453078, 0.92054737],
               [0.80587711, 0.39291452, 0.41042919],
               [0.24859572, 0.8812071 , 0.26392735],
               [0.74615831, 0.23623819, 0.75454159],
               [0.44253238, 0.34735206, 0.65710831],
               [0.9450387 , 0.51954284, 0.16603643]])
        >>> DigitalNetB2(dimension=3,randomize='LMS_DS',seed=5,alpha=2)(8)
        array([[0.08990191, 0.0928071 , 0.19312192],
               [0.75904054, 0.84222408, 0.97090093],
               [0.44323452, 0.88388537, 0.02322647],
               [0.64118039, 0.13329341, 0.79709894],
               [0.30380964, 0.22698085, 0.15664954],
               [0.59603521, 0.9775792 , 0.88242588],
               [0.17069108, 0.81102096, 0.11170129],
               [0.99563393, 0.061628  , 0.83357156]])
        >>> DigitalNetB2(dimension=3,randomize='LMS',seed=5,alpha=2)(8,warn=False)
        array([[0.        , 0.        , 0.        ],
               [0.83330849, 0.75137017, 0.78900948],
               [0.40020937, 0.95905275, 0.20511411],
               [0.699824  , 0.21041396, 0.99021704],
               [0.35457947, 0.17909801, 0.09927936],
               [0.56088333, 0.92774329, 0.8148018 ],
               [0.23708574, 0.84517128, 0.17932156],
               [0.91360582, 0.09382525, 0.89093793]])
        >>> DigitalNetB2(dimension=3,randomize='False',seed=5,alpha=2)(8,warn=False)
        array([[0.      , 0.      , 0.      ],
               [0.75    , 0.75    , 0.75    ],
               [0.4375  , 0.9375  , 0.1875  ],
               [0.6875  , 0.1875  , 0.9375  ],
               [0.296875, 0.171875, 0.109375],
               [0.546875, 0.921875, 0.859375],
               [0.234375, 0.859375, 0.171875],
               [0.984375, 0.109375, 0.921875]])
        >>> DigitalNetB2(dimension=3,randomize='NUS',seed=5,alpha=2)(8)
        array([[0.13470966, 0.19119158, 0.96130884],
               [0.97413754, 0.90343374, 0.02884907],
               [0.26275262, 0.86277465, 0.75808903],
               [0.59591889, 0.0264989 , 0.21845493],
               [0.49745711, 0.1158892 , 0.92224289],
               [0.65012418, 0.75808168, 0.09617135],
               [0.11594365, 0.98245561, 0.850095  ],
               [0.7761354 , 0.16906349, 0.15959343]])
        >>> DigitalNetB2(dimension=3,randomize='LMS_DS',seed=7,replications=2)(4)
        array([[[0.47687816, 0.88660568, 0.87000753],
                [0.71221591, 0.31703292, 0.32253034],
                [0.18809137, 0.0626922 , 0.09592865],
                [0.98586502, 0.64009274, 0.58067792]],
        <BLANKLINE>
               [[0.24653277, 0.1821862 , 0.74732591],
                [0.85333655, 0.52953251, 0.15818487],
                [0.47750995, 0.90831912, 0.42649986],
                [0.62241467, 0.25651801, 0.97657906]]])
        >>> DigitalNetB2(dimension=3,randomize='LMS',seed=7,replications=2)(4,warn=False)
        array([[[0.        , 0.        , 0.        ],
                [0.79796457, 0.70241587, 0.5475088 ],
                [0.28983202, 0.94903657, 0.7742629 ],
                [0.52522336, 0.254479  , 0.28954498]],
        <BLANKLINE>
               [[0.        , 0.        , 0.        ],
                [0.8961262 , 0.66083813, 0.59054646],
                [0.27006262, 0.77399226, 0.82226327],
                [0.62613416, 0.43372985, 0.27077997]]])
        >>> DigitalNetB2(dimension=3,randomize='DS',seed=7,replications=2)(4)
        array([[[0.04386058, 0.58727432, 0.3691824 ],
                [0.54386058, 0.08727432, 0.8691824 ],
                [0.29386058, 0.33727432, 0.6191824 ],
                [0.79386058, 0.83727432, 0.1191824 ]],
        <BLANKLINE>
               [[0.65212985, 0.69669968, 0.10605352],
                [0.15212985, 0.19669968, 0.60605352],
                [0.90212985, 0.44669968, 0.85605352],
                [0.40212985, 0.94669968, 0.35605352]]])
        >>> DigitalNetB2(dimension=3,randomize='OWEN',seed=7,replications=2)(4)
        array([[[0.35967208, 0.89954733, 0.20817919],
                [0.98911419, 0.49964829, 0.62480864],
                [0.02600317, 0.16945373, 0.97946126],
                [0.56008744, 0.54552841, 0.30711945]],
        <BLANKLINE>
               [[0.00782812, 0.71181447, 0.44244982],
                [0.53880628, 0.41910168, 0.65404289],
                [0.29051333, 0.16752862, 0.80878263],
                [0.77449486, 0.83546482, 0.22950922]]])
    
    **References:**

    1.  Marius Hofert and Christiane Lemieux (2019). 
        qrng: (Randomized) Quasi-Random Number Generators. 
        R package version 0.0-7.
        https://CRAN.R-project.org/package=qrng.

    2.  Faure, Henri, and Christiane Lemieux. 
        “Implementation of Irreducible Sobol' Sequences in Prime Power Bases.” 
        Mathematics and Computers in Simulation 161 (2019): 13-22. Crossref. Web.

    3.  F.Y. Kuo & D. Nuyens.
        Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
        - a survey of analysis and implementation, Foundations of Computational Mathematics, 
        16(6):1631-1696, 2016.
        springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
        arxiv link: https://arxiv.org/abs/1606.06613
        
    4.  D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/
        https://bitbucket.org/dnuyens/qmc-generators/src/cb0f2fb10fa9c9f2665e41419097781b611daa1e/cpp/digitalseq_b2g.hpp

    5.  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. 
        (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. 
        In H. Wallach, H. Larochelle, A. Beygelzimer, F. d extquotesingle Alch&#39;e-Buc, E. Fox, & R. Garnett (Eds.), 
        Advances in Neural Information Processing Systems 32 (pp. 8024-8035). Curran Associates, Inc. 
        Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

    6.  I.M. Sobol', V.I. Turchaninov, Yu.L. Levitan, B.V. Shukhman: 
        "Quasi-Random Sequence Generators" Keldysh Institute of Applied Mathematics, 
        Russian Academy of Sciences, Moscow (1992).

    7.  Sobol, Ilya & Asotsky, Danil & Kreinin, Alexander & Kucherenko, Sergei. (2011). 
        Construction and Comparison of High-Dimensional Sobol' Generators. Wilmott. 
        2011. 10.1002/wilm.10056. 

    8.  Paul Bratley and Bennett L. Fox. 1988. 
        Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
        ACM Trans. Math. Softw. 14, 1 (March 1988), 88-100. 
        DOI:https://doi.org/10.1145/42288.214372
    """

    def __init__(self,
                 dimension = 1,
                 replications = None,
                 seed = None,
                 randomize = 'LMS_DS',
                 generating_matrices = "joe_kuo.6.21201.txt",
                 order = 'NATURAL',
                 t = 63,
                 alpha = 1,
                 msb = None,
                 _verbose = False,
                 # deprecated
                 graycode = None,
                 t_max = None,
                 t_lms = None):
        r"""
        Args:
            dimension (Union[int,np.ndarray]): dimension of the generator.

                - If an `int` is passed in, use generating vector components at indices 0,...,`dimension`-1
                - If an `np.ndarray` is passed in, use generating vector components at these indices
            
            replications (int): number of IID randomizations of a pointset
            seed (Union[None,int,np.random.SeedSeq): seed the random number generator for reproducibility
            randomize (str): Options are
                
                - `"LMS_DS"`: Linear matrix scramble with digital shift
                - `"LMS"`: Linear matrix scramble only
                - `"DS"`: Digital shift only
                - `"NUS"`: Nested uniform scrambling. Also known as Owen scrambling. 
                - `"FALSE"` No randomization. In this case the first point will be the origin. 

            generating_matrices (Union[str,np.ndarray,int]: Specify the generating matrices.
                
                - A string `generating_matrices` should be the name (or path) of a file from the LDData repo at https://github.com/QMCSoftware/LDData/tree/main/dnet
                - An `np.ndarray` of integers with shape $(d,m_\mathrm{max})$ where $d$ is the number of dimensions.
                    Must supply `m_max` where $2^{m_\mathrm{max}}$ is the max number of supported samples. 
                    Must supply `t_max` which specifies the number of bits in each integer of the generating matrix.
                    May set `msb=False` 
            
            order (str): "NATURAL", or "GRAY" ordering.
            m_max (int): $2^{m_\mathrm{max}}$ is the max number of supported samples
            t (int): number of bits in integer represetation of points *after* randomization. The number of bits in the generating matrix is inferred
            alpha (int): interlacing factor for higher order nets. 
                WARNING: When alpha>1, interlacing is performed regardless of the generating matrices 
                i.e. for alpha>1 do NOT pass in generating matrices which are already interlaced
            msb (bool): if True, integers in generating matrices are stored in Most Significant Bit (MSB) order.
                If False, integers in geenrating matrices are stored in Least Significant Bit (LSB) order and must be bit-reversed. 
                For example, if `t_max = 3`, then 
                MSB order reads $6$ as $(1 1 0)$ while 
                LSB order reads $6$ as $(0 1 1)$
            _verbose (bool): if True, print linear matrix scrambling matrices
        """
        if graycode is not None:
            order = 'GRAY' if graycode else 'NATURAL'
            warnings.warn("graycode argument deprecated, set order='GRAY' or order='NATURAL' instead. Using order='%s'"%order,ParameterWarning)
        if t_lms is not None:
            t = t_lms
            warnings.warn("t_lms argument deprecated. Set t instead. Using t = %d"%t,ParameterWarning)
        if t_max is not None: 
            warnings.warn("t_max is deprecated as it can be inferred from the generating matrices. Set t to change the number of bits after randomization.",ParameterWarning)
        self.parameters = ['randomize','gen_mats_source','order','t','alpha','n_limit']
        self.input_generating_matrices = deepcopy(generating_matrices)
        self.input_t = deepcopy(t) 
        self.input_msb = deepcopy(msb)
        if isinstance(generating_matrices,str):
            self.gen_mats_source = generating_matrices
            assert generating_matrices[-4:]==".txt"
            local_root = dirname(abspath(__file__))+'/generating_matrices/'
            repos = DataSource()
            if repos.exists(local_root+generating_matrices):
                datafile = repos.open(local_root+generating_matrices)
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices)
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"+generating_matrices):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/"+generating_matrices)
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices[7:]):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/LDData/refs/heads/main/dnet/"+generating_matrices[7:])
            elif repos.exists("https://raw.githubusercontent.com/QMCSoftware/"+generating_matrices):
                datafile = repos.open("https://raw.githubusercontent.com/QMCSoftware/"+generating_matrices)
            elif repos.exists(generating_matrices):
                datafile = repos.open(generating_matrices)
            else:
                raise ParameterError("LDData path %s not found"%generating_matrices)
            contents = [line.rstrip('\n').strip() for line in datafile.readlines()]
            contents = [line.split("#",1)[0] for line in contents if line[0]!="#"]
            datafile.close()
            msb = True
            assert int(contents[0])==2, "DigitalNetB2 requires base=2 " # base 2
            d_limit = int(contents[1])
            n_limit = int(contents[2])
            self._t_curr = int(contents[3])
            compat_shift = self._t_curr-64 if self._t_curr>=64 else 0
            if compat_shift>0: warnings.warn("Truncating ints in generating matrix to have 64 bits.",ParameterWarning)
            gen_mats = np.array([[int(v)>>compat_shift for v in line.split(' ')] for line in contents[4:]],dtype=np.uint64)[None,:]
        elif isinstance(generating_matrices,np.ndarray):
            self.gen_mats_source = "custom"
            assert generating_matrices.ndim==2 or generating_matrices.ndim==3
            gen_mats = generating_matrices[None,:,:] if generating_matrices.ndim==2 else generating_matrices
            assert isinstance(msb,bool), "when generating_matrices is a np.ndarray you must set either msb=True (for most significant bit ordering) or msb=False (for least significant bit ordering which will require a bit reversal)"
            gen_mat_max = gen_mats.max() 
            assert gen_mat_max>0, "generating matrix must have positive ints"
            self._t_curr = int(np.ceil(np.log2(gen_mat_max+1)))
            d_limit = gen_mats.shape[1]
            n_limit = int(2**(gen_mats.shape[2]))
        else:
            raise ParameterError("invalid generating_matrices, must be a string or np.ndarray.")
        super(DigitalNetB2,self).__init__(dimension,replications,seed,d_limit,n_limit)
        assert gen_mats.ndim==3 and gen_mats.shape[1]>=self.d and (gen_mats.shape[0]==1 or gen_mats.shape[0]==self.replications) and gen_mats.shape[2]>0, "invalid gen_mats.shape = %s"%str(gen_mats.shape)
        self.m_max = int(gen_mats.shape[-1])
        if isinstance(generating_matrices,np.ndarray) and msb:
            qmctoolscl.dnb2_gmat_lsb_to_msb(np.uint64(gen_mats.shape[0]),np.uint64(self.d),np.uint64(self.m_max),np.tile(np.uint64(self._t_curr),int(gen_mats.shape[0])),gen_mats,gen_mats,backend="c")
        self.order = str(order).upper()
        assert self.order in ['NATURAL','GRAY']
        assert isinstance(t,int) and t>0
        assert self._t_curr<=t<=64, "t must no more than 64 and no less than %d (the number of bits used to represent the generating matrices)"%(self._t_curr)
        assert isinstance(alpha,int) and alpha>0
        self.alpha = alpha
        if self.alpha>0:
            assert (self.dvec==np.arange(self.d)).all(), "digital interlacing requires dimension is an int"
            if self.m_max!=self._t_curr:
                warnings.warn("Digital interlacing is often performed on matrices with the number of columns (m_max = %d) equal to the number of bits in each int (%d), but this is not the case. Ensure you are NOT setting alpha>1 when generating matrices are already interlaced."%(self.m_max,self._t_curr),ParameterWarning)
        self._verbose = _verbose
        self.randomize = str(randomize).upper()
        if self.randomize=="TRUE": self.randomize = "LMS_DS"
        if self.randomize=="OWEN": self.randomize = "NUS"
        if self.randomize=="NONE": self.randomize = "FALSE"
        if self.randomize=="NO": self.randomize = "FALSE"
        assert self.randomize in ["LMS_DS","LMS","DS","NUS","FALSE"]
        self.dtalpha = self.alpha*self.d
        if self.randomize=="FALSE":
            if self.alpha==1:
                self.gen_mats = gen_mats[:,self.dvec,:]
                self.t = self._t_curr
            else: 
                t_alpha = min(self.alpha*self._t_curr,64)
                gen_mat_ho = np.empty((gen_mats.shape[0],self.d,self.m_max),dtype=np.uint64)
                qmctoolscl.dnb2_interlace(np.uint64(gen_mats.shape[0]),np.uint64(self.d),np.uint64(self.m_max),np.uint64(self.dtalpha),np.uint64(self._t_curr),np.uint64(t_alpha),np.uint64(self.alpha),gen_mats[:,:self.dtalpha,:].copy(),gen_mat_ho,backend="c")
                self.gen_mats = gen_mat_ho
                self._t_curr = t_alpha
                self.t = self._t_curr
        elif self.randomize=="DS":
            if self.alpha==1:
                self.gen_mats = gen_mats[:,self.dvec,:]
                self.t = t
            else: 
                t_alpha = min(self.alpha*self._t_curr,64)
                gen_mat_ho = np.empty((gen_mats.shape[0],self.d,self.m_max),dtype=np.uint64)
                qmctoolscl.dnb2_interlace(np.uint64(gen_mats.shape[0]),np.uint64(self.d),np.uint64(self.m_max),np.uint64(self.dtalpha),np.uint64(self._t_curr),np.uint64(t_alpha),np.uint64(self.alpha),gen_mats[:,:self.dtalpha,:].copy(),gen_mat_ho,backend="c")
                self.gen_mats = gen_mat_ho
                self._t_curr = t_alpha
                self.t = t
            self.rshift = qmctoolscl.random_tbit_uint64s(self.rng,self.t,(self.replications,self.d))
        elif self.randomize in ["LMS","LMS_DS"]:
            if self.alpha==1:
                gen_mat_lms = np.empty((self.replications,self.d,self.m_max),dtype=np.uint64)
                S = qmctoolscl.dnb2_get_linear_scramble_matrix(self.rng,np.uint64(self.replications),np.uint64(self.d),np.uint64(self._t_curr),np.uint64(t),np.uint64(self._verbose))
                qmctoolscl.dnb2_linear_matrix_scramble(np.uint64(self.replications),np.uint64(self.d),np.uint64(self.m_max),np.uint64(gen_mats.shape[0]),np.uint64(t),S,gen_mats[:,self.dvec,:].copy(),gen_mat_lms,backend="c")
                self.gen_mats = gen_mat_lms
                self._t_curr = t
                self.t = self._t_curr
            else:
                gen_mat_lms = np.empty((self.replications,self.dtalpha,self.m_max),dtype=np.uint64)
                S = qmctoolscl.dnb2_get_linear_scramble_matrix(self.rng,np.uint64(self.replications),np.uint64(self.dtalpha),np.uint64(self._t_curr),np.uint64(t),np.uint64(self._verbose))
                qmctoolscl.dnb2_linear_matrix_scramble(np.uint64(self.replications),np.uint64(self.dtalpha),np.uint64(self.m_max),np.uint64(gen_mats.shape[0]),np.uint64(t),S,gen_mats[:,:self.dtalpha,:].copy(),gen_mat_lms,backend="c")
                gen_mat_lms_ho = np.empty((self.replications,self.d,self.m_max),dtype=np.uint64)
                qmctoolscl.dnb2_interlace(np.uint64(self.replications),np.uint64(self.d),np.uint64(self.m_max),np.uint64(self.dtalpha),np.uint64(t),np.uint64(t),np.uint64(self.alpha),gen_mat_lms,gen_mat_lms_ho,backend="c")
                self.gen_mats = gen_mat_lms_ho
                self._t_curr = t
                self.t = self._t_curr
            if self.randomize=="LMS_DS":
                self.rshift = qmctoolscl.random_tbit_uint64s(self.rng,self.t,(self.replications,self.d))
        elif self.randomize=="NUS":
            if alpha==1:
                new_seeds = self._base_seed.spawn(self.replications*self.d)
                self.rngs = np.array([np.random.Generator(np.random.SFC64(new_seeds[j])) for j in range(self.replications*self.d)]).reshape(self.replications,self.d)
                self.root_nodes = np.array([qmctoolscl.NUSNode_dnb2() for i in range(self.replications*self.d)]).reshape(self.replications,self.d)
                self.gen_mats = gen_mats[:,self.dvec,:].copy()
                self.t = t
            else:
                self.dtalpha = self.alpha*self.d
                new_seeds = self._base_seed.spawn(self.replications*self.dtalpha)
                self.rngs = np.array([np.random.Generator(np.random.SFC64(new_seeds[j])) for j in range(self.replications*self.dtalpha)]).reshape(self.replications,self.dtalpha)
                self.root_nodes = np.array([qmctoolscl.NUSNode_dnb2() for i in range(self.replications*self.dtalpha)]).reshape(self.replications,self.dtalpha)
                self.gen_mats = gen_mats[:,:self.dtalpha,:].copy()
                self.t = t
        else:
            raise ParameterError("self.randomize parsing error")
        gen_mat_max = self.gen_mats.max() 
        assert gen_mat_max>0, "generating matrix must have positive ints"
        assert self._t_curr==int(np.ceil(np.log2(gen_mat_max+1)))
        assert 0<self._t_curr<=self.t<=64, "invalid 0 <= self._t_curr (%d) <= self.t (%d) <= 64"%(self._t_curr,self.t)
        if self.randomize=="FALSE": assert self.gen_mats.shape[0]==self.replications, "randomize='FALSE' but replications = %d does not equal the number of sets of generating matrices %d"%(self.replications,self.gen_mats.shape[0])
    
    def _gen_samples(self, n_min, n_max, return_unrandomized, return_binary, warn):
        if n_min == 0 and self.randomize in ["FALSE","LMS"] and warn:
            warnings.warn("Without randomization, the first digtial net point is the origin",ParameterWarning)
        r_x = np.uint64(self.gen_mats.shape[0]) 
        n = np.uint64(n_max-n_min)
        d = np.uint64(self.dtalpha) if self.randomize=="NUS" and self.alpha>1 else np.uint64(self.d)
        n_start = np.uint64(n_min)
        mmax = np.uint64(self.m_max)
        xb = np.empty((r_x,n,d),dtype=np.uint64)
        if self.order=="GRAY":
            qmctoolscl.dnb2_gen_gray(r_x,n,d,n_start,mmax,self.gen_mats,xb,backend="c")
        elif self.order=="NATURAL": 
            assert (n_min==0 or np.log2(n_min)%1==0) and (n_max==0 or np.log2(n_max)%1==0), "DigitalNetB2 in natural order requires n_min and n_max be 0 or powers of 2"
            qmctoolscl.dnb2_gen_natural(r_x,n,d,n_start,mmax,self.gen_mats,xb,backend="c")
        else:
            "invalid digital net order" 
        r = np.uint64(self.replications)
        if "NUS" in self.randomize:
            if self.alpha==1:
                xrb = np.empty((r,n,d),dtype=np.uint64)
                qmctoolscl.dnb2_nested_uniform_scramble(r,n,d,r_x,np.uint64(self._t_curr),np.uint64(self.t),self.rngs,self.root_nodes,xb,xrb)
                xb = xrb
            else:
                d = np.uint64(self.d)
                dtalpha = np.uint64(self.dtalpha)
                t_alpha = self.t
                alpha = np.uint64(self.alpha)
                xrb_lo = np.empty((r,n,self.dtalpha),dtype=np.uint64)
                _ = qmctoolscl.dnb2_nested_uniform_scramble(r,n,dtalpha,r_x,np.uint64(self._t_curr),np.uint64(t_alpha),self.rngs,self.root_nodes,xb,xrb_lo)
                xrb_t = np.empty((r,d,n),dtype=np.uint64)
                xrb_lo_t = np.moveaxis(xrb_lo,[1,2],[2,1]).copy() 
                qmctoolscl.dnb2_interlace(r,d,n,dtalpha,np.uint64(t_alpha),np.uint64(self.t),alpha,xrb_lo_t,xrb_t)
                xb = np.moveaxis(xrb_t,[1,2],[2,1]).copy()
        if "DS" in self.randomize:
            xrb = np.empty((r,n,d),dtype=np.uint64)
            lshifts = np.tile(np.uint64((self.t-self._t_curr) if "LMS" not in self.randomize else 0),int(r))
            qmctoolscl.dnb2_digital_shift(r,n,d,r_x,lshifts,xb,self.rshift,xrb,backend="c")
            xb = xrb
        if return_binary:
            return xb
        else:
            x = np.empty((r,n,d),dtype=np.float64)
            tmaxes_new = np.tile(self.t,int(r)).astype(np.uint64)
            qmctoolscl.dnb2_integer_to_float(r,n,d,tmaxes_new,xb,x,backend="c")
            if return_unrandomized:
                return x,xb 
            else:
                return x
    
    def _spawn(self, child_seed, dimension):
        return DigitalNetB2(
            dimension = dimension,
            replications = None if self.no_replications else self.replications,
            seed = child_seed,
            randomize = self.randomize,
            generating_matrices = self.input_generating_matrices,
            order = self.order,
            t = self.input_t,
            alpha = self.alpha,
            msb = self.input_msb,
            _verbose = False,
            # deprecated
            graycode = None,
            t_max = None,
            t_lms = None)


class Sobol(DigitalNetB2): pass

