from .preconditioners import _BasePrecond,IdentityPrecond,PPCholPrecond,JacobiPrecond,SSORPrecond,BlockPrecond
from ..util import _get_npt
import numpy as np 
import time
import scipy.linalg
import itertools

def pcg(mat, b, precond=None, x0=None, rtol=None, atol=None, maxiter=None, beta_method=None, ref_sol=None, ckwargs={}):
    """
    >>> rng = np.random.Generator(np.random.PCG64(7)) 
    >>> n = 4
    >>> L = np.tril(rng.uniform(size=(n,n)))
    >>> L 
    array([[0.62509547, 0.        , 0.        , 0.        ],
           [0.30016628, 0.87355345, 0.        , 0.        ],
           [0.79706943, 0.46793495, 0.30303243, 0.        ],
           [0.25486959, 0.44507631, 0.50454826, 0.55349735]])
    >>> A = L@L.T
    >>> A
    array([[0.39074434, 0.18763258, 0.49824449, 0.15931782],
           [0.18763258, 0.85319542, 0.64801956, 0.4653012 ],
           [0.49824449, 0.64801956, 0.94611145, 0.56431   ],
           [0.15931782, 0.4653012 , 0.56431   , 0.82397969]])
    >>> b = rng.uniform(size=(n,))
    >>> b 
    array([0.99550028, 0.79266192, 0.62217923, 0.98896015])
    >>> xhat,data = pcg(A,b,rtol=0.,atol=0.)
    >>> x = scipy.linalg.cho_solve((L,True),b)
    >>> xhat-x
    array([ 3.55271368e-15, -2.66453526e-14, -6.75015599e-14, -2.84217094e-14])
    >>> data["rbackward_norms"]
    array([1.00000000e+00, 4.29203172e-01, 7.18633337e-01, 1.99316856e-01,
           8.67932537e-14])
    
    >>> xhat,data = pcg(A,b,rtol=0.,atol=0.,ref_sol=x)
    >>> data["rforward_norms"]
    array([1.00000000e+00, 9.87169026e-01, 6.26795202e-01, 3.03568508e-02,
           2.68397083e-15])
    
    >>> import torch
    >>> n = 25
    >>> L = np.tril(rng.uniform(size=(n,n)))
    >>> A = L@L.T
    >>> b = rng.uniform(size=(n,))
    >>> x = scipy.linalg.cho_solve((L,True),b)
    >>> Atorch = torch.from_numpy(A)
    >>> btorch = torch.from_numpy(b)
    >>> xtorch = torch.from_numpy(x)
    >>> for _A,_b,_x in [(A,b,x),(Atorch,btorch,xtorch)]:
    ...     for precond in [IdentityPrecond(),PPCholPrecond(_A),JacobiPrecond(_A),SSORPrecond(_A),BlockPrecond(_A,[0,len(_A)//2,len(_A)])]:
    ...         xhat,data = pcg(_A,_b,precond=precond,rtol=0.,atol=0.,ref_sol=_x)
    ...         print("%s: rerror = %.1e"%(type(precond).__name__,data["rforward_norms"][-1].item()))
    ...         print(precond.info_str(_A,spaces=4))  
    IdentityPrecond: rerror = 1.0e+00
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        2.3e+10        1.0e+00        
    PPCholPrecond: rerror = 8.3e-09
        K(A)           K(P)           K(P)/K(A)      Lk.shape       
        2.3e+10        2.9e+01        1.3e-09        (25, 24)       
    JacobiPrecond: rerror = 1.0e+00
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        1.1e+09        4.7e-02        
    SSORPrecond: rerror = 3.3e-01
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        6.0e+09        2.6e-01        
    BlockPrecond: rerror = 1.3e-09
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        2.2e+09        9.6e-02        
    IdentityPrecond: rerror = 1.0e+00
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        2.3e+10        1.0e+00        
    PPCholPrecond: rerror = 1.6e-09
        K(A)           K(P)           K(P)/K(A)      Lk.shape       
        2.3e+10        2.9e+01        1.3e-09        (25, 24)       
    JacobiPrecond: rerror = 1.0e+00
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        1.1e+09        4.7e-02        
    SSORPrecond: rerror = 3.3e-01
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        6.0e+09        2.6e-01        
    BlockPrecond: rerror = 3.2e-09
        K(A)           K(P)           K(P)/K(A)      
        2.3e+10        2.2e+09        9.6e-02        

    Preconditioned Conjugate Gradient (PCG) method, 
    see https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    
    Args:
        mat (function): matrix A
        b (np.ndarray or torch.Tensor): right hand side 
        precond (_BasePreconditioner subclass): preconditioner Px = b for x where P is the preconditioner. 
        x0 (np.ndarray or torch.Tensor): initial guess
        rtol, atol (float): require norm(b-A@x)<=max(rtol*norm(b),atol), defaults to rtol,atol = 1e-8,0
        maxiter (int): maximum number of iterations 
        beta_method (str): either 'FR' for Fletcher-Reeves (default) or 'PR' for Polak-Ribiere
        ref_sol (np.ndarray or torch.Tensor): reference solution for computing forward error
        npt (numpy or torch): module for numerical computations 
        ckwargs (dict): keyword arguments to pass when constructing a tensor e.g. ckwargs={"device":"cpu"}
    
    Returns: 
        x (np.ndarray or torch.Tensor): approximate solution 
        data (dict): containing values
            x (np.ndarray or torch.Tensor): approximate solution
            rbackward_norms (np.ndarray or torch.Tensor): relative backwards norms ||b_hat - b||_2 / ||b||_2
            times (np.ndarray or torch.Tensor): times elapsed for each iteration using time.perf_counter
            rforward_norms (np.ndarray or torch.Tensor): relative forward norms || x_hat - x ||_2 / ||x||_2, 
                only set if ref_sol is provided
    """
    npt = _get_npt(mat)
    if rtol is None: rtol = 1e-8 
    if atol is None: atol = 0.
    if beta_method is None: beta_method = "FR"
    assert isinstance(beta_method,str) and beta_method.upper() in ['FR','PR']
    beta_method = beta_method.upper()
    n = len(b)
    if maxiter is None: maxiter = n 
    assert 0<maxiter<=n
    assert b.shape==(n,)
    bnorm = npt.linalg.norm(b)
    x = npt.zeros(n,dtype=float,**ckwargs) if x0 is None else x0
    assert x.shape==(n,)
    assert rtol>=0 and atol>=0
    residtol = max(rtol*bnorm,atol)
    if precond is None: precond = IdentityPrecond()
    assert isinstance(precond,_BasePrecond)
    backward_norms = npt.zeros(maxiter+1,dtype=float,**ckwargs)
    if ref_sol is not None:
        xrefnorm = npt.linalg.norm(ref_sol)
        forward_norms = npt.zeros(maxiter+1,dtype=float,**ckwargs)
        forward_norms[0] = npt.linalg.norm(ref_sol) if x0 is None else npt.linalg.norm(ref_sol-x)
    times = npt.zeros(maxiter+1,dtype=float,**ckwargs)
    t0 = time.perf_counter()
    r = b if x0 is None else b-mat@x
    z = precond.solve(r)
    rz = r@z
    p = z
    backward_norms[0] = npt.linalg.norm(r)
    for i in range(1,maxiter+1):
        Ap = mat@p
        alpha = rz/(p@Ap)
        x = x+alpha*p
        r = r-alpha*Ap
        backward_norms[i] = npt.linalg.norm(r)
        if ref_sol is not None:
            forward_norms[i] = npt.linalg.norm(x-ref_sol)
        times[i] = time.perf_counter()-t0
        if backward_norms[i]<=residtol or i==maxiter: break
        znew = precond.solve(r)
        rznew = r@znew
        if beta_method=="FR":
            beta = rznew/rz 
        else: #beta_method=="PR"
            beta = r@(znew-z)/rz
        z = znew
        rz = rznew
        p = z+beta*p
    data = {
        "x": x,
        "rbackward_norms": backward_norms[:(i+1)]/bnorm,
        "times": times[:(i+1)]}
    if ref_sol is not None:
        data["rforward_norms"] = forward_norms[:(i+1)]/xrefnorm
    return x,data
    