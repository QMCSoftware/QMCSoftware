import numpy as np 
import scipy.linalg

def ppchol(A, rank=None, rtol=None, atol=None, return_pivots=False):
    """
    Partial pivoted Cholesky decomposition 
    https://www.sciencedirect.com/science/article/abs/pii/S0168927411001814

    >>> n = 10 
    >>> L = np.tril(np.random.rand(n,n))
    >>> A = L@L.T 
    >>> Lk = ppchol(A,rank=5,rtol=0,atol=0)
    >>> import torch 
    >>> At = torch.from_numpy(A)
    >>> Lkt = ppchol(At,rank=5,rtol=0.,atol=0.) 
    >>> import gpytorch
    >>> Lkgpt = gpytorch.pivoted_cholesky(At,rank=5)
    >>> assert (Lk==Lkgpt.numpy()).all()
    >>> assert (Lkt==Lkgpt).all()

    Arg:
        A (np.ndarray or torch.Tensor): n x n matrix 
        rank (int): rank of partial Cholesky decomposition 
        rtol,atol (float): stop when diag(A-Ak)<=max(atol,rtol*diag(A)) 
            where Ak = Lk@Lk.T and L_k is the n x k partial poivoted cholesky decomposition 
        return_pivots (bool): flag to return pivots 

    Returns:
        n x k np.ndarray or torch.Tensor Lk
    """
    if isinstance(A,np.ndarray):
        npt = np
        clone = lambda x: x.copy()
    else: 
        import torch 
        npt = torch
        clone = lambda x: x.clone()
    if rtol is None: rtol = 1e-10
    if atol is None: atol = 0.
    n = len(A)
    assert A.shape==(n,n)
    d = clone(A.diagonal())
    traceA = d.sum() 
    tol = max(atol,rtol*traceA) 
    error = traceA
    pi = npt.arange(n)
    if rank is None: rank = n
    Lkt = npt.zeros((rank,n),dtype=A.dtype)
    for m in range(rank):
        i = d[pi[m:]].argmax().item()+m
        pi[[m,i]] = pi[[i,m]]
        pim = pi[m].item()
        Lkt[m,pim] = npt.sqrt(d[pim])
        piis = pi[(m+1):]
        Lkt[m,piis] = (A[pim,piis] - (Lkt[:m,[pim]]*Lkt[:m,piis]).sum(0))/Lkt[m,pim]
        d[piis] = d[piis]-Lkt[m,piis]**2
        error = d[pi[(m+1):]].sum()
        if error<=tol: break 
    Lkt = Lkt[:(m+1)]
    if return_pivots:
        return Lkt.T,pi
    else:
        return Lkt.T

def solve_ppchol(b, Lk, pL, ddiag):
    """
    Solve a system (torch.diag(ddiag)+Lk)@x = b for x 
    https://en.wikipedia.org/wiki/Woodbury_matrix_identity 
    https://arxiv.org/pdf/1809.11165

    >>> rng = np.random.Generator(np.random.PCG64(7))
    >>> n = 15
    >>> Ltrue = np.tril(rng.uniform(size=(n,n)))
    >>> ddiag = rng.uniform(size=(n,))
    >>> A = Ltrue@Ltrue.T
    >>> An = A+np.diag(ddiag)
    >>> Lnoisy = np.linalg.cholesky(An)
    >>> b = rng.uniform(size=(n,))
    >>> x = scipy.linalg.cho_solve((Lnoisy,True),b)
    >>> Lk = ppchol(A,rank=n,rtol=0.,atol=0.)
    >>> k = Lk.shape[1]
    >>> pL = np.linalg.cholesky(np.eye(k)+(Lk.T/ddiag)@Lk)
    >>> xhat = solve_ppchol(b,Lk,pL,ddiag)
    >>> assert np.allclose(x,xhat,atol=1e-12)
    >>> import torch
    >>> xhatt = solve_ppchol(torch.from_numpy(b),torch.from_numpy(Lk),torch.from_numpy(pL),torch.from_numpy(ddiag))
    >>> assert np.allclose(x,xhatt.numpy(),atol=1e-12)
    >>> Lk = ppchol(A,rank=n-1) 
    >>> k = Lk.shape[1]
    >>> pL = np.linalg.cholesky(np.eye(k)+(Lk.T/ddiag)@Lk)
    >>> xhat = solve_ppchol(b,Lk,pL,ddiag) 
    >>> assert np.linalg.norm(xhat-x)/np.linalg.norm(x)<1e-10

    Args:
        b (np.ndarray or torch.Tensor): n x p matrix or length n vector RHS 
        Lk (np.ndarray or torch.Tensor): shape n x k partial pivoted Cholesky decomposition 
        L (np.ndarray or torch.Tensor): Cholesky factor of (Ik+Lk.T/ddiag)@Lk where Ik is the k x k identity matrix 
        diag (np.ndarray or torch.Tensor): length n vector of diagonal entries 

    Returns:
        np.ndarray or torch.Tensor linear system solution 
    """
    if isinstance(b,np.ndarray):
        cho_solve = lambda x,L: scipy.linalg.cho_solve((L,True),x)
    else: 
        import torch 
        cho_solve = lambda x,L: torch.cholesky_solve(x,L)
    n = len(Lk)
    bcol = b.reshape((n,-1))
    x = bcol/ddiag[:,None]
    x = Lk.T@x
    x = cho_solve(x,pL)
    x = Lk@x
    x = bcol-x
    x /= ddiag[:,None]
    return x.reshape(b.shape)