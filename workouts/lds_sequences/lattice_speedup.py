from qmcpy import *
import itertools

import time

if __name__ == '__main__':

    n = 3000
    num_ports = 6000
    trials = 3

    for order, is_parallel in itertools.product(['natural', 'linear', 'MPS'], [True, False]):
        start_time = time.time()

        for trial in range(trials):
            l = Lattice(dimension=n, seed=42, order=order, is_parallel=is_parallel)
            weights = l.gen_samples(num_ports)  # using lattice points instead of iid

        end_time = time.time()

        print(f"{is_parallel = }, {trials = }, for {order = }, {end_time - start_time:.3f} s. {weights[1,5] = }\n")

"""

is_parallel = True, trials = 3, for order = 'natural', 1.896 s. weights[1,5] = 0.8235392637361244

is_parallel = False, trials = 3, for order = 'natural', 3.113 s. weights[1,5] = 0.8235392637361244

is_parallel = True, trials = 3, for order = 'linear', 3.242 s. weights[1,5] = 0.7970500059236244

is_parallel = False, trials = 3, for order = 'linear', 3.237 s. weights[1,5] = 0.7970500059236244

is_parallel = True, trials = 3, for order = 'MPS', 1.860 s. weights[1,5] = 0.8235392637361244

is_parallel = False, trials = 3, for order = 'MPS', 3.206 s. weights[1,5] = 0.8235392637361244

"""