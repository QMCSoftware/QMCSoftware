import numpy as np 
import scipy.special 
from . import ParameterError
from .torch_numpy_ops import get_npt

EPS64 = np.finfo(np.float64).eps
    
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

def parse_assign_param(pname, param, shape_param, requires_grad_param, tfs_param, endsize_ops, constraints, shape_batch, torchify, npt, nptkwargs):
    if np.isscalar(param):
        param = param*npt.ones(shape_param,**nptkwargs)
    else:
        if torchify:
            param = npt.atleast_1d(npt.tensor(param))
            assert isinstance(param,npt.Tensor), "%s must be a scalar or torch.Tensor"%pname
        else: 
            param = npt.atleast_1d(npt.array(param))
            assert isinstance(param,npt.ndarray), "%s must be a scalar or np.ndarray"%pname
    shape_param = list(param.shape)
    assert len(shape_param)>=1, "invalid shape_%s = %s"%(pname,str(shape_param))
    assert shape_param==(shape_batch+[shape_param[-1]])[-len(shape_param):], "incompatible shape_%s=%s and shape_batch=%s"%(pname,shape_param,shape_batch)
    assert len(tfs_param)==2, "tfs_scale should be a tuple of length 2"
    assert callable(tfs_param[0]), "tfs_scale[0] should be a callable e.g. torch.log"
    assert callable(tfs_param[1]), "tfs_scale[1] should be a callable e.g. torch.exp"
    raw_param = tfs_param[0](param)
    tf_param = tfs_param[1]
    if torchify:
        assert isinstance(requires_grad_param,bool)
        raw_param = npt.nn.Parameter(raw_param,requires_grad=requires_grad_param)
    assert shape_param[-1] in endsize_ops, "%s not in %s"%(str(shape_param[-1]),str(endsize_ops))
    if "POSITIVE" in constraints: 
        assert (param>0).all(), "%s must be positive"%pname
    if "INTEGER" in constraints:
        assert (param%1==0).all(), "%s must be integers"%pname
    return raw_param,tf_param
