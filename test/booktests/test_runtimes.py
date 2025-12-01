"""
Estimated runtimes for notebook tests (in seconds).
Based on sequential execution measurements.
Used for Longest Job First (LJF) and optimal bin-packing scheduling.

To update these values, run the sequential tests and extract timing info:
    python -m unittest discover -s . -p "tb_*.py" -v

Tests are sorted by runtime (longest first) for optimal load balancing.
"""
import heapq

# Estimated runtimes in seconds (from sequential run on 2025-12-01)
# Sorted from longest to shortest for easy reference
TEST_RUNTIMES = {
    # Long-running tests (> 60s) - these are the bottleneck
    'tb_iris': 138.10,
    'tb_elliptic_pde': 96.12,
    'tb_qei_demo_for_blog': 88.40,
    'tb_asian_option_mlqmc': 81.97,
    'tb_vectorized_qmc_bayes': 69.83,
    
    # Medium-running tests (20-60s)
    'tb_digital_net_b2': 41.37,
    'tb_ray_tracing': 38.57,
    'tb_gaussian_diagnostics_demo': 35.80,
    'tb_lattice_random_generator': 18.54,
    'tb_linear_scrambled_halton': 17.88,
    'tb_MCQMC_2020_QMC_Software_Tutorial': 16.94,
    'tb_vectorized_qmc': 16.89,
    'tb_plot_proj_function': 14.29,
    'tb_joss2025': 12.93,
    'tb_control_variates': 10.81,
    'tb_pricing_options': 10.43,
    
    # Short-running tests (< 20s)
    'tb_sample_scatter_plots': 8.16,
    'tb_nei_demo': 7.58,
    'tb_some_true_measures': 7.18,
    'tb_why_add_q_to_mc_blog': 4.73,
    'tb_lebesgue_integration': 4.10,
    'tb_qmcpy_intro': 4.01,
    'tb_quickstart': 3.81,
    'tb_gbm_demo': 1.82,
    'tb_acm_toms_sorokin_2025': 1.76,
    'tb_umbridge': 0.85,
    
    # Skipped tests (assigned small default time)
    'tb_Argonne_2023_Talk_Figures': 1.0,
    'tb_MCQMC2022_Article_Figures': 1.0,
    'tb_Purdue_Talk_Figures': 1.0,
    'tb_dakota_genz': 1.0,
    'tb_prob_failure_gp_ci': 1.0,
    'tb_pydata_chi_2023': 1.0,
}

# Default runtime for unknown tests (assume medium-length)
DEFAULT_RUNTIME = 30.0


def get_runtime(test_name: str) -> float:
    """Get estimated runtime for a test module."""
    return TEST_RUNTIMES.get(test_name, DEFAULT_RUNTIME)


def sort_by_runtime(test_modules: list, longest_first: bool = True) -> list:
    """
    Sort test modules by estimated runtime (simple LJF).
    
    Args:
        test_modules: List of test module names
        longest_first: If True, sort longest jobs first (LJF scheduling)
        
    Returns:
        Sorted list of test module names
    """
    return sorted(
        test_modules,
        key=lambda m: get_runtime(m),
        reverse=longest_first
    )


def optimal_schedule(test_modules: list, num_workers: int) -> tuple:
    """
    Compute optimal job scheduling using LPT (Longest Processing Time) algorithm.
    This is a greedy bin-packing approach that assigns each job to the least loaded worker.
    
    Args:
        test_modules: List of test module names
        num_workers: Number of parallel workers
        
    Returns:
        tuple: (ordered_modules, worker_assignments, estimated_makespan)
            - ordered_modules: Tests ordered for optimal submission
            - worker_assignments: List of (worker_id, [test_names]) 
            - estimated_makespan: Estimated parallel execution time
    """
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    
    # Sort by runtime (longest first) - LPT heuristic
    sorted_modules = sort_by_runtime(test_modules, longest_first=True)
    
    # Min-heap of (current_load, worker_id, [assigned_tests])
    workers = [(0.0, i, []) for i in range(num_workers)]
    heapq.heapify(workers)
    
    # Assign each job to the least loaded worker
    for module in sorted_modules:
        runtime = get_runtime(module)
        # Pop the least loaded worker
        current_load, worker_id, assigned = heapq.heappop(workers)
        # Assign this job
        assigned.append(module)
        new_load = current_load + runtime
        # Push back with updated load
        heapq.heappush(workers, (new_load, worker_id, assigned))
    
    # Extract results
    worker_assignments = [(w_id, tests) for (load, w_id, tests) in sorted(workers, key=lambda x: x[1])]
    worker_loads = [(w_id, sum(get_runtime(t) for t in tests)) for (w_id, tests) in worker_assignments]
    estimated_makespan = max(load for (_, load) in worker_loads) if worker_loads else 0
    
    # Create optimal submission order: interleave from each worker's queue
    # to maximize parallelism from the start
    ordered_modules = []
    queues = [list(tests) for (_, tests) in worker_assignments]
    while any(queues):
        for q in queues:
            if q:
                ordered_modules.append(q.pop(0))
    
    return ordered_modules, worker_assignments, estimated_makespan


def print_schedule(test_modules: list, num_workers: int = None) -> None:
    """Print the scheduled order with estimated runtimes."""
    if num_workers is None:
        # Just print LJF order
        sorted_modules = sort_by_runtime(test_modules, longest_first=True)
        print("\nScheduled test order (Longest Job First):")
        print("-" * 50)
        total_time = 0
        for i, module in enumerate(sorted_modules, 1):
            runtime = get_runtime(module)
            total_time += runtime
            print(f"{i:3}. {module:45} {runtime:7.2f}s")
        print("-" * 50)
        print(f"Total estimated sequential time: {total_time:.2f}s")
    else:
        # Print optimal schedule for given workers
        ordered, assignments, makespan = optimal_schedule(test_modules, num_workers)
        total_seq = sum(get_runtime(m) for m in test_modules)
        
        print(f"\n{'='*60}")
        print(f"OPTIMAL SCHEDULE FOR {num_workers} WORKERS")
        print(f"{'='*60}")
        print(f"Total sequential time: {total_seq:.2f}s")
        print(f"Estimated parallel time (makespan): {makespan:.2f}s")
        print(f"Theoretical speedup: {total_seq/makespan:.2f}x")
        print(f"Efficiency: {(total_seq/makespan)/num_workers*100:.1f}%")
        print(f"{'='*60}")
        
        for worker_id, tests in assignments:
            load = sum(get_runtime(t) for t in tests)
            print(f"\nWorker {worker_id} (load: {load:.2f}s, {len(tests)} tests):")
            for t in tests:
                print(f"  - {t}: {get_runtime(t):.2f}s")
        
        print(f"\n{'='*60}")


def analyze_scalability(test_modules: list, max_workers: int = 16) -> dict:
    """
    Analyze how speedup scales with different worker counts.
    
    Args:
        test_modules: List of test module names  
        max_workers: Maximum number of workers to analyze
        
    Returns:
        dict: Scalability analysis results
    """
    total_seq = sum(get_runtime(m) for m in test_modules)
    longest_job = max(get_runtime(m) for m in test_modules)
    
    print(f"\n{'='*70}")
    print("SCALABILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total sequential time: {total_seq:.2f}s")
    print(f"Longest single job: {longest_job:.2f}s (hard floor for parallel time)")
    print(f"{'='*70}")
    print(f"{'Workers':>8} | {'Makespan':>10} | {'Speedup':>8} | {'Efficiency':>10} | {'vs Ideal':>10}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")
    
    results = {}
    for n in range(1, max_workers + 1):
        _, _, makespan = optimal_schedule(test_modules, n)
        speedup = total_seq / makespan
        efficiency = speedup / n * 100
        ideal_speedup = min(n, total_seq / longest_job)
        vs_ideal = speedup / ideal_speedup * 100
        print(f"{n:>8} | {makespan:>10.2f} | {speedup:>8.2f} | {efficiency:>9.1f}% | {vs_ideal:>9.1f}%")
        results[n] = {'makespan': makespan, 'speedup': speedup, 'efficiency': efficiency}
    
    print(f"{'='*70}")
    print(f"Note: Maximum theoretical speedup = {total_seq/longest_job:.2f}x (limited by longest job)")
    print()
    
    return results


if __name__ == '__main__':
    # Demo: analyze scalability for all tests
    all_tests = list(TEST_RUNTIMES.keys())
    analyze_scalability(all_tests, max_workers=12)
    
    # Show optimal schedule for 4 and 8 workers
    print_schedule(all_tests, num_workers=4)
    print_schedule(all_tests, num_workers=8)
