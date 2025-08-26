import numpy as np 

def get_npt(x):
    if isinstance(x,np.ndarray):
        return np
    else:
        import torch
        assert isinstance(x,torch.Tensor)
        return torch