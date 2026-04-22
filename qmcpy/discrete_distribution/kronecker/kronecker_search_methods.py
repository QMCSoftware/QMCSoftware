import numpy as np
from sympy import gcdex, primerange, prime
#https://github.com/sympy/sympy/releases

# I can't find where Jimmy's code for the kronecker search from SURE 2025 is, so I've temporarily put my method here
# Currently, this produces results that are very similar but not identical to my matlab code, which is a bit concerning.
# The wssd of the two methods are typically the same to 2-3 decimal places, depending on N and d

def KTildeEx(t):
    b = t * (t - 1) + 1/6
    step = 1 + b * (1 / (np.arange(1, len(t) + 1) ** 2))
    k = np.prod(step)
    return k

def kronecker_search_march_2026(N, dMax, searchsize):
    if searchsize < 2:
        raise ValueError("searchsize must be at least 2.")
    if N < 2:
        raise ValueError("N must be at least 2.")
    if dMax < 1:
        raise ValueError("dMax must be at least 1.")
    
    # search over the first n primes, n = searchsize
    searchspace = np.array(list(primerange(1, prime(searchsize)+1)))

    alpha = np.zeros(dMax)
    # we pick the golden ratio as the first alpha
    alpha[0] = (np.sqrt(5) - 1) / 2
    alpha[0] = (np.sqrt(5) - 1) / 2
    
    diff = np.cumsum(1.0 / np.arange(N, 1, -1))
    freq = np.cumsum(diff)
    freq = np.flip(freq)
    
    # Compute Bezout coefficients for all pairs in the search space
    bezoutCoeffs = np.zeros((searchsize, searchsize))
    for i in range(searchsize - 1):
        a = searchspace[i]
        for j in range(i + 1, searchsize):
            c = searchspace[j]
            # Use sympy.gcdex to get Bezout coefficients
            d_coeff, b_coeff, _ = gcdex(int(a), int(c))
            bezoutCoeffs[i, j] = b_coeff
            bezoutCoeffs[j, i] = d_coeff
    bezoutCoeffs = np.abs(bezoutCoeffs)
    
    # setting up some useful variables for the search
    coeff = np.zeros((dMax - 1, 4))
    num = N * (N + 1) / 2
    t = np.mod(alpha[0] * np.arange(1,N), 1)
    kPrev = 1 + (t * (t - 1) + 1/6)
    
    # the main search loop
    for dim in range(1, dMax):
        best = np.array([0, 0, 0, 0, np.inf])
        nK0 = N * KTildeEx(np.zeros(dim+1))
        for i in range(searchsize):
            p1 = searchspace[i]
            for j in range(searchsize):
                if j == i:
                    continue
                p2 = searchspace[j]
                b = bezoutCoeffs[i, j]
                d = bezoutCoeffs[j, i]
                alpha_dim = (p1 * alpha[dim - 1] + b) / (p2 * alpha[dim - 1] + d)
                t = (alpha_dim * np.arange(1, N)) - np.floor(alpha_dim * np.arange(1, N))
                k_vector = kPrev * (1 + (t * (t - 1) + 1/6) / ((dim+1) ** 2))
                
                wssd = nK0 - num + 2 * np.dot(freq, k_vector)
                
                if wssd < best[4]:
                    best[0] = p1
                    best[1] = b
                    best[2] = p2
                    best[3] = d
                    best[4] = wssd
        
        alpha_d = (best[0] * alpha[dim - 1] + best[1]) / (best[2] * alpha[dim - 1] + best[3])
        alpha[dim] = np.mod(alpha_d, 1)
        t = np.mod(alpha[dim] * np.arange(1, N), 1)
        kPrev = kPrev * (1 + (t * (t - 1) + 1/6) / ((dim+1) ** 2))
        coeff[dim - 1, :] = [best[0], best[1], best[2], best[3]]
        #print(coeff[dim - 1, :], best[4]) #debugging line to check the coefficients and wssd at each dimension
    
    return alpha

# quick and dirty test
# print(kronecker_search_march_2026(10000, 20, 50))