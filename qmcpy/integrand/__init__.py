from .asian_option import AsianOption
from .european_option import EuropeanOption
from .keister import Keister
from .linear import Linear
from .custom_fun import CustomFun
from .ml_call_options import MLCallOptions

import numpy as np

# computes the periodization transform for the given function values
def do_period_transform(f_input, ptransform):
    if ptransform == 'Baker':
        f = lambda x: f_input(1 - 2 * abs(x - 1 / 2))  # Baker's transform
    elif ptransform == 'C0':
        f = lambda x: f_input(3 * x ** 2 - 2 * x ** 3) * np.prod(6 * x * (1 - x), 1)  # C^0 transform
    elif ptransform == 'C1':
        # C^1 transform
        f = lambda x: f_input(x ** 3 * (10 - 15 * x + 6 * x ** 2)) * np.prod(30 * x ** 2 * (1 - x) ** 2, 1)
    elif ptransform == 'C1sin':
        # Sidi C^1 transform
        f = lambda x: f_input(x - np.sin(2 * np.pi * x) / (2 * np.pi)) * np.prod(2 * np.sin(np.pi * x) ** 2, 1)
    elif ptransform == 'C2sin':
        # Sidi C^2 transform
        psi3 = lambda t: (8 - 9 * np.cos(np.pi * t) + np.cos(3 * np.pi * t)) / 16
        psi3_1 = lambda t: (9 * np.sin(np.pi * t) * np.pi - np.sin(3 * np.pi * t) * 3 * np.pi) / 16
        f = lambda x: f_input(psi3(x)) * np.prod(psi3_1(x), 1)
    elif ptransform == 'C3sin':
        # Sidi C^3 transform
        psi4 = lambda t: (12 * np.pi * t - 8 * np.sin(2 * np.pi * t) + np.sin(4 * np.pi * t)) / (12 * np.pi)
        psi4_1 = lambda t: (12 * np.pi - 8 * np.cos(2 * np.pi * t) * 2 * np.pi + np.sin(
            4 * np.pi * t) * 4 * np.pi) / (12 * np.pi)
        f = lambda x: f_input(psi4(x)) * np.prod(psi4_1(x), 1)
    elif ptransform == 'none':
        # do nothing
        f = lambda x: f_input(x)
    else:
        f = f_input
        print(f'Error: Periodization transform {ptransform} not implemented')

    return f
