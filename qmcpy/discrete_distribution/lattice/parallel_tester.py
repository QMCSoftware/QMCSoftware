from qmcpy import Lattice

import cProfile
import pstats

#I only uncomment the code statement that includes what I want to test, everything else I leave commented.


#Magic Point Shop
'''
if __name__ == "__main__":
    input_x = 1_000_000
    process = 0
    nonprocess = 0
    c = 5
    print(f"For input size {input_x}")
    print(f'Each funtion was ran {c} times')
    for i in range(c):
        profiler = cProfile.Profile()
        profiler.enable()
        Lattice(dimension=2, randomize=False, order="mps").gen_samples(input_x, warn=False)
        profiler.disable()
        stats = pstats.Stats(profiler)
        nonprocess_time = stats.total_tt
        print(f"Cumulative Time: for non-process Magic point shop: {nonprocess_time:.2f} seconds")
        profiler1 = cProfile.Profile()
        profiler1.enable()
        Lattice(dimension=2, randomize=False, order="mps_process").gen_samples(input_x, warn=False)
        profiler1.disable()
        stats1 = pstats.Stats(profiler1)
        process_time = stats1.total_tt
        print(f"Cumulative Time: for Process Magic point shop {process_time:.2f} seconds")
        process += process_time
        nonprocess += nonprocess_time
    print(f"Process {process}")
    print(f"NonProcess {nonprocess}")
    if process < nonprocess:
        print(f"Process is this much better: {nonprocess - process}")
    else:
        print(f"Non-process is this much better: {process - nonprocess}")
'''

#Natural order

'''
if __name__ == "__main__":
    input_x = 1_000_000
    process = 0
    nonprocess = 0
    c = 30
    print(f"For input size {input_x}")
    print(f'There are {c} function calls')
    for i in range(c):
        profiler = cProfile.Profile()
        profiler.enable()
        Lattice(dimension=2, randomize=False, order="natural").gen_samples(input_x, warn=False)
        profiler.disable()

        stats = pstats.Stats(profiler)

        # Get the cumulative time for the entire program
        nonprocess_time = stats.total_tt
        print(f"Cumulative Time: for non-process natural: {nonprocess_time:.2f} seconds")

        profiler1 = cProfile.Profile()
        profiler1.enable()
        Lattice(dimension=2, randomize=False, order="natural_process").gen_samples(input_x, warn=False)
        profiler1.disable()

        stats1 = pstats.Stats(profiler1)
        # Get the cumulative time for the entire program
        process_time = stats1.total_tt
        print(f"Cumulative Time: for Process natural {process_time:.2f} seconds")

        process += process_time
        nonprocess += nonprocess_time
    print(f"Process {process}")
    print(f"Non-process {nonprocess}")
    if process < nonprocess:
        print(f"Process is this much better: {nonprocess - process}")
    else:
        print(f"Non-process is this much better: {process - nonprocess}")

# Linear Order
'''
'''
if __name__ == "__main__":
    input_x = 1_000_000
    process = 0
    nonprocess = 0
    c = 15
    print(f"For input size {input_x}")
    print(f"The functions were called {c} times")
    for i in range(c):
        profiler = cProfile.Profile()
        profiler.enable()
        Lattice(dimension=2, randomize=False, order="linear").gen_samples(input_x, warn=False)
        profiler.disable()

        stats = pstats.Stats(profiler)

        # Get the cumulative time for the entire program
        nonprocess_time = stats.total_tt
        print(f"Cumulative Time: for non-process linear: {nonprocess_time:.2f} seconds")

        profiler1 = cProfile.Profile()
        profiler1.enable()
        Lattice(dimension=2, randomize=False, order="linear_process").gen_samples(input_x, warn=False)
        profiler1.disable()

        stats1 = pstats.Stats(profiler1)
        # Get the cumulative time for the entire program
        process_time = stats1.total_tt
        print(f"Cumulative Time: for Process linear {process_time:.2f} seconds")

        process += process_time
        nonprocess += nonprocess_time
    print(f"Process {process}")
    print(f"Non-Process {nonprocess}")
    if process < nonprocess:
        print(f"Process is this much better: {nonprocess - process}")
    else:
        print(f"Non-process is this much better: {process - nonprocess}")
        
'''



'''
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    l = Lattice(dimension=1, order="mps_process")
    print(l.gen_samples(4))
    '''



if __name__ == "__main__":
    x = Lattice(dimension=1, order="mps_process")
    print(x.gen_samples(7))


