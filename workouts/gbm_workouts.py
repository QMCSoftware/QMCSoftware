import matplotlib.pyplot as plt
from parsl_config import *
from qmcpy import GeometricBrownianMotion
from qmcpy import DigitalNetB2
import numpy as np
import time

def main():
    # Create an instance of GeometricBrownianMotion
    gbm = GeometricBrownianMotion(DigitalNetB2(100, seed=7), t_final=2, drift=0.1, diffusion=0.2)

    # Define the number of samples to generate
    num_samples = 2 ** 18

    # Split the number of samples into smaller chunks
    chunk_size = 2 ** 14
    num_chunks = num_samples // chunk_size

    # Generate samples sequentially
    sequential_start = time.time()
    sequential_samples = gbm.gen_samples(num_samples)
    sequential_end = time.time()
    sequential_time = sequential_end - sequential_start

    # Generate samples using Parsl (since each GBM path can be simulated independently of others)
    parallel_start = time.time()
    futures = []
    for _ in range(num_chunks):
        future = generate_samples(gbm, chunk_size)
        futures.append(future)

    # Wait for all the futures to complete
    results = [future.result() for future in futures]

    # Concatenate the results
    parallel_samples = np.concatenate(results)
    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start


    print(f"Generated {len(sequential_samples)} samples sequentially in {sequential_time:.3f} seconds.")
    print(f"Generated {len(parallel_samples)} samples using Parsl in {parallel_time:.3f} seconds.")
    print(f"Speedup: {sequential_time / parallel_time:.3f}x")



if __name__ == "__main__":
    main()

"""Output:
Generated 262144 samples sequentially in 1.266 seconds.
Generated 262144 samples using Parsl in 0.357 seconds.
Speedup: 3.545x
"""