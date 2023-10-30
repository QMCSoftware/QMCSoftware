import concurrent.futures
import pstats
import cProfile

def square(number):
    return number * number

def fib(n):
    if n == 1:
        return 1
    if n == 0:
        return 0
    return fib(n-1) + fib(n-2)

def main():
    run_list = [x for x in range(20)]

    profiler = cProfile.Profile()
    profiler.enable()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_list = list(executor.map(fib, run_list))

    profiler.disable()
    profiler.print_stats("cumulative")

    stats = pstats.Stats(profiler)
    stats_dict = stats.stats


    function_stats = dict()

    # Iterate over the statistics and store them in an array
    for func_info, func_stats in stats.stats.items():
        func_name = func_info[2]
        cumulative_time = func_stats[3]
        function_stats[round(cumulative_time,5)] = func_name



    sorted_dict = {k: v for k, v in sorted(function_stats.items())}
    print(sorted_dict)


'''
    profiler = cProfile.Profile()
    profiler.enable()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        result_list = list(executor.map(fib, run_list))

    profiler.disable()
    profiler.print_stats("cumulative")

    p = pstats.Stats(profiler)
    total_runtime = p.total_tt
'''






if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
