from ctypes import CDLL, RTLD_GLOBAL
from os.path import dirname, abspath
from glob import glob

c_lib = CDLL(glob(dirname(abspath(__file__))+'/c_lib*')[0], mode=RTLD_GLOBAL)
