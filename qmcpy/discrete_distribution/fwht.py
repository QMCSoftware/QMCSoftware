
from ..util import ParameterError, ParameterWarning
from .c_lib import c_lib
import ctypes
from numpy import *

class FWHT():
    def __init__(self):
        self.fwht_copy_cf = c_lib.fwht_copy
        self.fwht_copy_cf.argtypes = [
            ctypes.c_uint32,
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')
        ]
        self.fwht_copy_cf.restype = None

        self.fwht_inplace_cf = c_lib.fwht_inplace
        self.fwht_inplace_cf.argtypes = [
            ctypes.c_uint32,
            ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        ]
        self.fwht_inplace_cf.restype = None

    def fwht_copy(self, n, src, dst):
        self.fwht_copy_cf(n, src, dst)

    def fwht_inplace(self, n, src):
        self.fwht_inplace_cf(n, src)