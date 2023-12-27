import qmcpy as qp


def mps(n, num_ports, seed=None):
    l = qp.Lattice(dimension=n, seed=seed,order="mps" ,is_parallel=False)
    points = l.gen_samples(num_ports)
    return points

def mps_thread(n, num_ports, seed=None):
    l = qp.Lattice(dimension=n, seed=seed,order="mps" ,is_parallel=True)
    points = l.gen_samples(num_ports)
    return points


d_list = [50, 100, 200, 400, 800, 1600]
num_ponts = 10
thread = []
non_thread = []



for i in d_list:
    mps(i , num_ponts)
    mps_thread(i, num_ponts)



