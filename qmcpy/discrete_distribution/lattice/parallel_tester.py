from qmcpy import Lattice
import time



def run_three_func(n):
    start_1 = time.time()

    l = Lattice(2, seed=7, order="natural")
    l.gen_samples(n)

    end_1 = time.time()

    start_2 = time.time()

    l = Lattice(2, seed=7, order="linear")
    l.gen_samples(n)

    end_2 = time.time()

    start_3 = time.time()

    l = Lattice(2, seed=7, order="mps")

    l.gen_samples(n)

    end_3 = time.time()

    return end_1-start_1, end_2-start_2, end_3-start_3

time1, time2, time3 = run_three_func(100)

print(f"Time for natural {time1}")
print(f"Time for linear {time2}")
print(f"Time for mps {time3}")


