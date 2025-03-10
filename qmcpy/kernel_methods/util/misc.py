import numpy as np 

def _get_npt(mat):
    try:
        return mat.npt
    except: 
        pass 
    if isinstance(mat,np.ndarray):
        return np
    else:
        try:
            import torch
            assert isinstance(mat,torch.Tensor)
        except:
            raise Exception("invalid mat, must be a _GramMatrix, _PDEGramMatrix, np.ndarray or torch.Tensor")
        return torch