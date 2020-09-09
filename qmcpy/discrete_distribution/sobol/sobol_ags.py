from ...util import ParameterError, ParameterWarning
from ..c_lib import c_lib
import ctypes
from os.path import dirname, abspath, isfile
from numpy import *
import warnings


class SobolAGS(object):
    """ A custom base 2 Sobol' generator by alegresor """

    def __init__(self, dimension, randomize, graycode, seed, z_path=None, d0=0):
        # initialize c code
        self.sobol_ags_cf = c_lib.sobol_ags
        self.sobol_ags_cf.argtypes = [
            ctypes.c_ulong,  # n
            ctypes.c_uint32,  # d
            ctypes.c_ulong, # n0
            ctypes.c_uint32, # d0
            ctypes.c_uint8,  # randomize
            ctypes.c_uint8, # graycode
            ctypeslib.ndpointer(ctypes.c_uint32, flags='C_CONTIGUOUS'), # seeds
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),  # x (result)
            ctypes.c_uint32, # d_max
            ctypes.c_uint32, # m_max
            ctypeslib.ndpointer(ctypes.c_ulong, flags='C_CONTIGUOUS'),  # z (generating matrix)
            ctypes.c_uint8] # msb
        errors = {
            1: 'requires 32 bit precision but system has unsigned int with < 32 bit precision.',
            2: 'using natural ordering (graycode=0) where n0 and/or (n0+n) is not 0 or a power of 2 is not allowed.',
            3: 'using n0+n exceeds 2^m_max or d0+d exceeds d_max.'}
        # set parameters
        self.sobol_ags_cf.restype = ctyps.c_uint8
        self.set_dimension(dimension)
        self.set_seed(self.s_og)
        self.set_randomize(randomize)
        self.set_graycode(graycode)
        self.set_d0(d0)
        # set generating matrix
        if not z_path:
            self.d_max = 21201
            self.m_max = 32
            self.msb = True
            self.z = load(dirname(abspath(__file__))+'generating_matricies/gen_mat.21201.32.msb.npy')
        else:
            if not isfile(z_path):
                raise ParameterError('z_path `' + z_path + '` not found. ')
            self.z = load(z_path)
            f = z_path.split('/')[-1]
            f_lst = f.split('.')
            self.d_max = int(f_lst[1])
            self.m_max = int(f_lst[2])
            msblsb = f_lst[3].lower()
            if msblsb == 'msb':
                self.msb = True
            elif msblsb == 'lsb':
                self.msb = False
            else:
                msg = '''
                    z_path sould be formatted like `gen_mat.21201.32.msb.npy`
                    with name.d_max.m_max.msb_or_lsb.npy
                '''
                raise ParameterError(msg)

    def gen_samples(self, n_min, n_max, warn):
        if len(self.s) != self.d:
            msg = '''
                dimension and length of seeds must match. 
                Try calling the set_seed method after resetting the dimension.
            '''
            raise ParameterError(msg)
        if n_min == 0 and self.r==False and warn:
            warnings.warn("Non-randomized AGS Sobol sequence includes the origin",ParameterWarning)
        n = int(n_max-n_min)
        x = zeros((self.d,n), dtype=double)
        self.sobol_ags_cf(n, self.d, int(n_min), self.d0, self.r, self.g, self.s, x, self.d_max, self.m_max, self.z, self.msb)
        return x

    def set_seed(self, seed):
        if seed and len(seed)==self.d:
            self.s = seed
        else:
            random.seed(seed)
            self.s = random.randint(2**32,size=self.d)
        return self.s
        
    def set_dimension(self, dimension):
        self.d = dimension
        return self.d
    
    def get_params(self):
        return self.d, self.r, self.g, self.s
    
    def set_randomize(self, randomize):
        randomize = randomize.upper()
        if randomize in ["LMS","LINEAR MATRIX SCRAMBLE"]:
            self.r = 1
        elif randomize in ["DS","DIGITAL SHIFT"]:
            self.r = 2
        else:
            msg = '''
                AGS Sobol' randomize should be either 
                    'LMS' for Linear Matrix Scramble or 
                    'DS' for Digital Shift. 
            '''
            raise ParameterError(msg)
    
    def set_graycode(self, graycode):
        self.g = graycode
    
    def set_d0(self, d0):
        self.d0 = d0
