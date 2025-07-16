import numpy as np 
import scipy.special 
from ...util import ParameterError

EPS64 = np.finfo(np.float64).eps

def get_npt(x):
    if isinstance(np.ndarray):
        return np
    else:
        import torch
        assert isinstance(x,torch.Tensor)
        return torch
    
def tf_exp(x):
    npt = get_npt(x)
    return npt.exp(x) 

def tf_exp_inv(x):
    npt = get_npt(x)
    return npt.log(x) 

def tf_exp_eps(x):
    return tf_exp(x)+EPS64 

def tf_exp_eps_inv(x):
    return tf_exp_inv(x-EPS64) 

def tf_square(x):
    return x**2 

def tf_square_inv(x):
    npt = get_npt(x)
    return npt.sqrt(x) 

def tf_square_eps(x):
    return tf_square(x)+EPS64 

def tf_square_eps_inv(x):
    return tf_square_inv(x-EPS64)

def tf_explinear(x):
    npt = get_npt(x)
    if npt==np:
        return -scipy.special.log_expit(-x)
    else:
        return -npt.nn.functional.logsigmoid(-x)

def tf_explinear_inv(x):
    npt = get_npt(x)
    return npt.where(x<34,npt.log(npt.expm1(x)),x)

def tf_explinear_eps(x):
    return tf_explinear(x)+EPS64

def tf_explinear_eps_inv(x):
    return tf_explinear_inv(x-EPS64)

def tf_identity(x):
    return x 

def parse_assign_param(self, pname, param, shape_param, requires_grad_param, tfs_param, endsize_ops, constraints, shape_batch, torchify, npt, nptkwargs):
    if np.isscalar(param):
        param = param*npt.ones(shape_param,**nptkwargs)
    else:
        if torchify:
            assert isinstance(param,npt.Tensor), "%s must be a scalar or torch.Tensor"%pname
        else: 
            assert isinstance(param,npt.ndarray), "%s must be a scalar or np.ndarray"%pname
    shape_param = list(shape_param.shape)
    assert len(shape_param)>=1
    assert shape_param[:-1]==shape_batch[-(len(shape_param)-1):]
    assert len(tfs_param)==2, "tfs_scale should be a tuple of length 2"
    assert callable(tfs_param[0]), "tfs_scale[0] should be a callable e.g. torch.log"
    assert callable(tfs_param[1]), "tfs_scale[1] should be a callable e.g. torch.exp"
    setattr(self,"raw_%s"%pname,tfs_param[0](param))
    setattr(self,"tf_%s"%pname,tfs_param[1])
    if torchify:
        assert isinstance(requires_grad_param,bool)
        setattr(self,"raw_%s"%pname,npt.nn.Parameter(getattr(self,"raw_%s"%pname),requires_grad=requires_grad_param))
    assert shape_param[-1] in endsize_ops
    if "POSITIVE" in constraints: 
        assert (param>0).all(), "%s must be positive"%pname
    if "INTEGER" in constraints:
        assert (param%1==0).all(), "%s must be integers"%pname
    else:
        raise ParameterError("invalid constraint for %s parameter"%pname)
