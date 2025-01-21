import numpy as np 

def train_val_split(R, to_split, val_frac=1/8, shuffle=True, rng_shuffle_seed=None):
    if isinstance(to_split[0],np.ndarray):
        npt = np 
    else:
        import torch 
        assert isinstance(to_split[0],torch.Tensor), "to_split list must contain np.ndarray or torch.Tensor items"
        npt = torch 
    if shuffle:
        rng = np.random.Generator(np.random.PCG64(rng_shuffle_seed))
        tv_idx = rng.permutation(R)
        if npt!=np: tv_idx = torch.from_numpy(tv_idx)
    else:
        tv_idx = npt.arange(R)
    n_train = R-int(val_frac*R)
    n_val = R-n_train
    tidx = tv_idx[:n_train]
    vidx = tv_idx[n_train:]
    splits = (vval for val in to_split for vval in (val[tidx],val[vidx]))
    return splits 