from ctypes import CDLL, RTLD_GLOBAL
from os.path import dirname, abspath
from glob import glob
import os

_c_lib = None
for file in glob(dirname(abspath(__file__)) + os.sep + "_c_lib*"):
    try:
        _c_lib = CDLL(file, mode=RTLD_GLOBAL)
        break
    except OSError:
        pass

assert isinstance(_c_lib,CDLL), "problem importing QMCPy' C library"