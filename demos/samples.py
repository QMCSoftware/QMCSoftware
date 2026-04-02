import multiprocess
import numpy as np 
import qmcpy as qp

def simulate(M_start,M_end,R_start,R_end,seed):
    print("M_start %-10d M_end %-10d R_start %-10d R_end %-10d seed %-10d"%(M_start,M_end,R_start,R_end,seed))
    ds_mmaxes = {
        1: 3,
        2: 3,
        4: 3,
        6: 3,
        }
    data = {}
    for i,(d,mmax) in enumerate(ds_mmaxes.items()):
        faure = qp.Faure(dimension=d,replications=int((M_end-M_start)*(R_end-R_start)),seed=seed,randomize="NUS",warn=False)
        p = int(faure.bases[0,0])
        n = p**mmax
        x = faure(n).reshape((M_end-M_start,R_end-R_start,n,d))
        data["d.%d.p.%d.mmax.%d.n.%d"%(d,p,mmax,n)] = x
    np.savez("M_start.%d.M_end.%d.R_start.%d.R_end.%d.npz"%(M_start,M_end,R_start,R_end),**data)
    print("Saved numpy file M_start.%d.M_end.%d.R_start.%d.R_end.%d.npz"%(M_start,M_end,R_start,R_end))

if __name__ == "__main__":
    parallel = True # set to True to enable multiprocessing 
    processes = 1 # number of processes
    global_seed = 7 # global seed 
    M = 10 # independent trials
    R = 100 # replications
    M_split = 5
    R_split = 50
    M_batch_size = int(np.ceil(M/M_split))
    R_batch_size = int(np.ceil(R/R_split))
    job_idx = 0 
    M_start = 0
    args = []
    while M_start<M:
        M_end = min(M_start+M_batch_size,M)
        R_start = 0
        while R_start<R:
            R_end = min(R_start+R_batch_size,R)
            seed = job_idx
            print("job_idx %-10d M_start %-10d M_end %-10d R_start %-10d R_end %-10d seed %-10d"%(job_idx,M_start,M_end,R_start,R_end,seed))
            assert R_start<R_end
            assert M_start<M_end
            args += [[M_start,M_end,R_start,R_end,seed]]
            job_idx += 1
            R_start = R_end 
        M_start = M_end
    if parallel:
        with multiprocess.Pool(processes=processes) as pool:
            pool.starmap(simulate,args)
    else:
        for i in range(len(args)):
            simulate(*args[i])
