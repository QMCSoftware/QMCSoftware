from ctypes import CDLL, RTLD_GLOBAL
from os.path import dirname, abspath
from glob import glob
import os

try:
    c_lib = CDLL(glob(dirname(abspath(__file__)) + os.sep + "c_lib*")[0], mode=RTLD_GLOBAL)
except Exception as e:
    pass