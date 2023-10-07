# !pip3 install parsl
import parsl
import time
from parsl.app.app import python_app
from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

print(parsl.__version__)

num_workers = 4

config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex",
            worker_debug=False,
            cores_per_worker=1,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=num_workers,  # number of initial workers
                max_blocks=num_workers,   # max number of workers
            ),
        )
    ],
    strategy=None,
)
parsl.load(config)


@python_app
def pi(num_points):
    from random import random
    inside = 0
    for i in range(num_points):
        x, y = random(), random()
        if x ** 2 + y ** 2 < 1:
            inside += 1
    return inside * 4 / num_points


@python_app
def mean(numbers):
    return sum(numbers) / len(numbers)


n = 100  # Increased number of tasks
num_pts = 10 ** 6
start_time = time.time()
# Estimate n values for pi with fewer points
pi_estimates = [pi(num_pts) for _ in range(n)]
# Batch all results
pi_results = [pi_estimate.result() for pi_estimate in pi_estimates]
print(f"{pi_results = }")
mean_pi = mean(pi_results)
print(f"Average: {mean_pi.result():.5f}")
end_time = time.time()
print(f"With parallelization: {end_time - start_time:.3} seconds")

###################################################
# Serial implementation
###################################################
def pi(num_points):
    from random import random
    inside = 0
    for i in range(num_points):
        x, y = random(), random()  # Drop a random point in the box.
        if x ** 2 + y ** 2 < 1:  # Count points within the circle.
            inside += 1
    return inside * 4 / num_points


def mean(numbers):
    return sum(numbers) / len(numbers)


start_time = time.time()
pi_estimates0 = [pi(num_pts) for _ in range(n)]
print(f"{pi_estimates0 = }")
mean_pi0 = mean(pi_estimates0)
print(f"Average: {mean_pi0:.5f}")
end_time = time.time()
print(f"Without parallelization: {end_time - start_time:.3} seconds")

"""
pi_results = [3.139008, 3.140692, 3.140516, 3.143616, 3.143008, 3.14238, 3.141752, 3.142716, 3.140932, 3.144604, 3.144904, 3.140632, 3.139944, 3.144256, 3.13808, 3.144196, 3.140056, 3.141148, 3.137812, 3.142116, 3.137792, 3.14292, 3.142744, 3.139956, 3.143456, 3.14206, 3.140744, 3.142352, 3.143528, 3.141796, 3.142252, 3.140832, 3.141104, 3.138324, 3.138568, 3.140684, 3.142556, 3.141304, 3.143648, 3.139056, 3.140916, 3.141628, 3.141628, 3.14004, 3.13912, 3.141092, 3.138968, 3.142272, 3.142776, 3.1403, 3.144312, 3.141164, 3.14282, 3.1436, 3.141744, 3.138924, 3.138436, 3.13994, 3.141976, 3.141612, 3.140228, 3.141184, 3.141124, 3.143632, 3.140204, 3.142068, 3.142264, 3.140124, 3.141936, 3.140812, 3.140296, 3.141316, 3.140044, 3.138612, 3.141656, 3.1433, 3.14296, 3.142328, 3.1431, 3.14266, 3.1429, 3.140644, 3.138836, 3.141536, 3.14224, 3.142696, 3.144576, 3.1396, 3.139136, 3.142668, 3.143976, 3.14146, 3.144092, 3.141824, 3.14152, 3.143012, 3.141576, 3.142172, 3.142208, 3.140888]
Average: 3.14149
With parallelization: 2.08 seconds
pi_estimates0 = [3.143568, 3.143516, 3.146488, 3.140996, 3.144284, 3.143328, 3.141248, 3.138752, 3.142272, 3.142324, 3.141144, 3.142144, 3.140332, 3.139228, 3.142956, 3.139668, 3.141104, 3.1405, 3.137572, 3.139064, 3.140376, 3.139672, 3.141908, 3.140352, 3.141732, 3.14192, 3.141396, 3.140316, 3.1382, 3.142764, 3.141284, 3.143212, 3.14372, 3.145344, 3.143672, 3.144188, 3.143888, 3.138852, 3.139032, 3.143788, 3.140992, 3.140708, 3.1434, 3.139552, 3.14156, 3.14096, 3.14244, 3.141804, 3.139176, 3.140296, 3.139576, 3.14164, 3.140772, 3.14244, 3.14188, 3.139512, 3.143208, 3.142608, 3.142564, 3.1442, 3.140672, 3.142336, 3.142944, 3.142904, 3.143124, 3.141076, 3.141308, 3.14312, 3.142752, 3.140228, 3.141612, 3.141236, 3.14244, 3.138972, 3.144376, 3.141032, 3.14282, 3.141468, 3.140708, 3.1417, 3.139624, 3.138816, 3.144308, 3.139164, 3.140052, 3.141544, 3.139848, 3.142664, 3.141508, 3.140892, 3.146844, 3.140136, 3.142144, 3.141484, 3.144652, 3.13844, 3.140876, 3.141724, 3.13968, 3.141984]
Average: 3.14159
Without parallelization: 12.8 seconds
"""