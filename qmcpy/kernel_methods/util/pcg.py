import numpy as np 
import time
import scipy.linalg

def pcg(matmul, b, x0=None, rtol=None, atol=None, maxiter=None, precond_solve=False, ref_sol=None, npt=np, ckwargs={}, beta_method="FR"):
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
    >>> xhat,rbackward_norms,times = pcg(lambda x: A@x,b,rtol=0.,atol=0.)
    >>> x = scipy.linalg.cho_solve((L,True),b)
    >>> xhat-x
    array([8.52651283e-14, 9.68114477e-14, 4.26325641e-14, 8.43769499e-14])
    >>> rbackward_norms
    array([1.00000000e+00, 4.29203172e-01, 7.18633337e-01, 1.99316856e-01,
           1.79201971e-13])
    >>> xhat,rbackward_norms,times,rforward_norms = pcg(lambda x: A@x,b,rtol=0.,atol=0.,ref_sol=x)
    >>> rforward_norms
    array([1.00000000e+00, 9.87169026e-01, 6.26795202e-01, 3.03568508e-02,
           5.50209863e-15])
    >>> import torch
    >>> Atorch = torch.from_numpy(A)
    >>> btorch = torch.from_numpy(b)
    >>> xtorch = torch.from_numpy(x)
    >>> xhat,rbackward_norms,times,rforward_norms = pcg(lambda x: Atorch@x,btorch,rtol=0.,atol=0.,ref_sol=xtorch, npt=torch)
    >>> xhat-xtorch
    tensor([9.2371e-14, 8.1712e-14, 3.5527e-15, 7.2831e-14], dtype=torch.float64)

    Preconditioned Conjugate Gradient (PCG) method, see https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
    
    Args:
        matmul (function): multiply Ax 
        b (np.ndarray or torch.Tensor): right hand side 
        x0 (np.ndarray or torch.Tensor): initial guess
        rtol, atol (float): require norm(b-A@x)<=max(rtol*norm(b),atol), defaults to rtol,atol = 1e-8,0
        maxiter (int): maximum number of iterations 
        precond_solve (function): function which solves the system Px = b for x where P is the preconditioner. 
        ref_sol (np.ndarray or torch.Tensor): reference solution for computing forward error
        npt (numpy or torch): module for numerical computations 
        ckwargs (dict): keyword arguments to pass when constructing a tensor e.g. ckwargs={"device":"cpu"}
        beta_method (str): either 'FR' for Fletcher-Reeves (default) or 'PR' for Polak-Ribiere
    """
    if rtol is None: rtol = 1e-8 
    if atol is None: atol = 0.
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
    assert precond_solve is False or callable(precond_solve)
    if precond_solve is False: 
        precond_solve = lambda r: r 
    backward_norms = npt.zeros(maxiter+1,dtype=float,**ckwargs)
    if ref_sol is not None:
        xrefnorm = npt.linalg.norm(ref_sol)
        forward_norms = npt.zeros(maxiter+1,dtype=float,**ckwargs)
        forward_norms[0] = npt.linalg.norm(ref_sol) if x0 is None else npt.linalg.norm(ref_sol-x)
    times = npt.zeros(maxiter+1,dtype=float,**ckwargs)
    t0 = time.perf_counter()
    r = b if x0 is None else b-matmul(x)
    z = precond_solve(r)
    rz = r@z
    p = z
    backward_norms[0] = npt.linalg.norm(r)
    for i in range(1,maxiter+1):
        Ap = matmul(p)
        alpha = rz/(p@Ap)
        x = x+alpha*p
        r = r-alpha*Ap
        backward_norms[i] = npt.linalg.norm(r)
        if ref_sol is not None:
            forward_norms[i] = npt.linalg.norm(x-ref_sol)
        times[i] = time.perf_counter()-t0
        if backward_norms[i]<=residtol or i==maxiter: break
        znew = precond_solve(r)
        rznew = r@znew
        if beta_method=="FR":
            beta = rznew/rz 
        else: #beta_method=="PR"
            beta = r@(znew-z)/rz
        z = znew
        rz = rznew
        p = z+beta*p
    times = times[:(i+1)]
    rbackward_norms = backward_norms[:(i+1)]/bnorm
    if ref_sol is None:
        return x,rbackward_norms,times
    else:
        rforward_norms = forward_norms[:(i+1)]/xrefnorm
        return x,rbackward_norms,times,rforward_norms