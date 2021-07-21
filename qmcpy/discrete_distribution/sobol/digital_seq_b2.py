"""
Digital Net Generator by alegresor 

References:
        
    [1] Paul Bratley and Bennett L. Fox. 1988. 
    Algorithm 659: Implementing Sobol's quasirandom sequence generator. 
    ACM Trans. Math. Softw. 14, 1 (March 1988), 88–100. 
    DOI:https://doi.org/10.1145/42288.214372
"""

from numpy import *

def _flip_bits(e, t_max):
    """
    flip the int e with t_max bits
    """
    u = 0
    for t in range(t_max):
        bit = array((1<<t),dtype=uint64)&e
        if bit:
            u += 1<<(t_max-t-1)
    return u

def _count_set_bits(e):
    """
    count the number of bits set to 1 in int e
    Brian Kernighan algorithm code: https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
    """
    if (e == 0): return 0
    else: return 1 + _count_set_bits(e&(e-1)) 
    
def _is_0_or_pow2(e):
    """
    check if e is 0 or a power of 2
    """
    return e==0 or e&(e-1)==0

def set_digitalnetb2_randomizations(dvec, d, set_lms, set_rshift, seeds, d_max, m_max, t_max, z, msb, print_scramble_mat=False):
    """
    initialize digital net generator in base 2 by
        - flipping least significant bit (lsb) order to most significant bit (msb) order
        - optionally applying a linear matrix scramble (lms) to the generating vector
        - optionally creating a vector of random (digital) shifts 
    Args:
        dvec (ndarray uint32): length d vector of dimesions
        d (int): length(dvec) = number of dimensions
        set_lms (bool): lms flag
        set_rshift (bool): random shift flag
        seeds (ndarray uint32): length d vector of seeds, one for each dimension in dvec
        d_max (uint32): max supported dimension 
        m_max (uint32): 2^m_max is the maximum number of samples supported
        t_max (uint32): number of bits in each element of z, the generating vector
        z (ndarray uint64): original generating vector with shape d_max x m_max
        msb (bool): msb flag  e.g. 6 is [1 1 0] in msb and [0 1 1] in lsb
        print_scramble_mat (bool): flag to print the scrambling matrix
    
    Return:
        znew (ndarray uint64): d x m_max generating vector in msb order, possibly with lms applied, for gen_digitalnetb2
        rshift (ndarray uint64): length d vector of random digital shifts for gen_digitalnetb2
    """
    # parameter checks
    if (dvec>d_max).any():
        raise Exception("require (dvec <= d_max).all()")
    if t_max > 64:
        raise Exception("require t_max <= 64")
    if m_max != t_max:
        raise Exception("higher order digital nets with m_max != t_max are not yet supported")
    # constants
    randomizer = random.RandomState()
    znew = zeros((d,m_max),dtype=uint64)
    rshift = zeros(d,dtype=uint64)
    zmsb = z[dvec,:]
    # flip bits if using lsb (least significant bit first) order
    if not msb:
        for j in range(d):
            for m in range(m_max):
                zmsb[j,m] = _flip_bits(zmsb[j,m],t_max)
    # set the linear matrix scrambling and random shift
    if set_lms and print_scramble_mat: print('s (scrambling_matrix)')
    for j in range(d):
        randomizer.seed(seeds[j])
        if set_lms:
            if print_scramble_mat: print('\n\ts[dvec[%d]]\n\t\t'%j,end='',flush=True)
            for t in range(t_max):
                u = randomizer.randint(low=0, high=1<<t, size=1, dtype=uint64)
                u <<= (t_max-t)
                u += 1<<(t_max-t-1)
                if print_scramble_mat:
                    for t1 in range(t_max):
                        mask = 1<<(t_max-t1-1)
                        bit = (u&mask)>0
                        print('%-2d'%bit,end='',flush=True)
                    print('\n\t\t',end='',flush=True)
                for m in range(m_max):
                    v = u&zmsb[j,m]
                    s = _count_set_bits(v)%2
                    if s: znew[j,m] += 1<<(t_max-t-1)
        else:
            znew = zmsb
        if set_rshift:
            rshift[j] = randomizer.randint(low=0, high=2**m_max, size=1, dtype=uint64)
    return znew,rshift

def gen_digitalnetb2(n0, n, d, graycode, m_max, znew, set_rshift, rshift):
    """
    generate samples from a digital net in base 2

    Args:
        n0 (uint32): starting index in sequence. Must be a power of 2 if not using Graycode ordering
        n (uint32): sample indicies include n0:(n0+n). Must be a power of 2 if not using Graycode ordering */
        d (uint32): dimension
        graycode (bool): Graycode ordering flag /* Graycode flag */
        m_max (uint32): 2^m_max is the maximum number of samples supported
        t_max (uint32): number of bits in each element of z, the generating vector
        znew (ndarray uint64): generating vector with shape d_max x m_max from set_digitalnetb2_randomizations
        set_rshift (bool): random shift flag
        rshift (ndarray uint64): length d vector of random digital shifts from set_digitalnetb2_randomizations
    
    Returns:
        x (ndarray float64): unrandomized samples with shape n x d
        xr (ndarray float64): randomized samples with shape n x d
    """
    # parameter checks
    if n==0 or d==0:
        return zeros(0,dtype=float64)
    if (not graycode) and not (_is_0_or_pow2(n0) and _is_0_or_pow2(n0+n)):
        raise Exception('''
        Natural ordering requires n0 and (n0+n) be either 0 or powers of 2.
        Use Graycode ordering for more flexible sequence indexing. ''')
    if n0+n > 2**m_max:
        raise Exception("sample index too large: n_max = n0+n is greater than 2^m_max")
    # constants
    scale = 2**(-m_max)
    x = zeros((n,d),dtype=float64)
    xr = zeros((n,d),dtype=float64)
    # generate points
    for j in range(d):
        # set an initial point 
        xj = array(0,dtype=uint64) # current point
        zj = array(0,dtype=uint64) # next directional vector
        if n0>0:
            im = n0-1
            b = im
            im ^= im>>1
            m = 0
            while im!=0 and m<m_max:
                if im&1:
                    xj ^= znew[j,m]
                im >>= 1
                m += 1
            s = 0
            while b&1:
                b >>= 1
                s += 1
            zj = znew[j,s]
        # set the rest of the points
        for i in range(n0,n0+n):
            xj ^= zj
            # set point
            im = i
            if not graycode:
                im = i^(i>>1)
            x[im-n0,j] = xj*scale
            if set_rshift:
                xr[im-n0,j] = (xj^rshift[j])*scale
            # get the index of the rightmost 0 bit in i
            b = i 
            s = 0
            while b&1:
                b >>= 1
                s += 1
            # get the vector used for the next index
            zj = znew[j,s]
    return x,xr

if __name__ == '__main__':
    # constants
    dvec = array([0,1,2],dtype=uint32)
    d = len(dvec)
    set_lms = False
    set_rshift = True
    seeds = [7,11,13]
    d_max = 3
    m_max = 32
    t_max = 32
    msb = False
    if msb:
        z = array([
            [2147483648     ,1073741824     ,536870912      ,268435456      ,134217728      ,67108864       ,33554432       ,16777216       ,8388608        ,4194304        ,2097152        ,1048576        ,524288         ,262144         ,131072         ,65536          ,32768          ,16384          ,8192           ,4096           ,2048           ,1024           ,512            ,256            ,128            ,64             ,32             ,16             ,8              ,4              ,2              ,1              ],
            [2147483648     ,3221225472     ,2684354560     ,4026531840     ,2281701376     ,3422552064     ,2852126720     ,4278190080     ,2155872256     ,3233808384     ,2694840320     ,4042260480     ,2290614272     ,3435921408     ,2863267840     ,4294901760     ,2147516416     ,3221274624     ,2684395520     ,4026593280     ,2281736192     ,3422604288     ,2852170240     ,4278255360     ,2155905152     ,3233857728     ,2694881440     ,4042322160     ,2290649224     ,3435973836     ,2863311530     ,4294967295     ],
            [2147483648     ,3221225472     ,1610612736     ,2415919104     ,3892314112     ,1543503872     ,2382364672     ,3305111552     ,1753219072     ,2629828608     ,3999268864     ,1435500544     ,2154299392     ,3231449088     ,1626210304     ,2421489664     ,3900735488     ,1556135936     ,2388680704     ,3314585600     ,1751705600     ,2627492864     ,4008611328     ,1431684352     ,2147543168     ,3221249216     ,1610649184     ,2415969680     ,3892340840     ,1543543964     ,2382425838     ,3305133397     ]],
            dtype=uint64)
    else:
        z = array([
            [1              ,2              ,4              ,8              ,16             ,32             ,64             ,128            ,256            ,512            ,1024           ,2048           ,4096           ,8192           ,16384          ,32768          ,65536          ,131072         ,262144         ,524288         ,1048576        ,2097152        ,4194304        ,8388608        ,16777216       ,33554432       ,67108864       ,134217728      ,268435456      ,536870912      ,1073741824     ,2147483648     ],
	        [1              ,3              ,5              ,15             ,17             ,51             ,85             ,255            ,257            ,771            ,1285           ,3855           ,4369           ,13107          ,21845          ,65535          ,65537          ,196611         ,327685         ,983055         ,1114129        ,3342387        ,5570645        ,16711935       ,16843009       ,50529027       ,84215045       ,252645135      ,286331153      ,858993459      ,1431655765     ,4294967295     ],
	        [1              ,3              ,6              ,9              ,23             ,58             ,113            ,163            ,278            ,825            ,1655           ,2474           ,5633           ,14595          ,30470          ,43529          ,65815          ,197434         ,394865         ,592291         ,1512982        ,3815737        ,7436151        ,10726058       ,18284545       ,54132739       ,108068870      ,161677321      ,370540567      ,960036922      ,2004287601     ,2863268003     ]],
            dtype=uint64)
    print_scramble_mat = True
    znew,rshift = set_digitalnetb2_randomizations(dvec, d, set_lms, set_rshift, seeds, d_max, m_max, t_max, z, msb, print_scramble_mat)
    n = 4
    n0 = 0
    graycode = True
    x,xr = gen_digitalnetb2(n0, n, d, graycode, m_max, znew, set_rshift, rshift)
    print("\n\nz")
    for j in range(d):
        dim = dvec[j]
        print('\n\tz[dvec[%d]]\n\t\t'%j,end='',flush=True)
        for t in range(t_max):
            mask = array(1<<(t_max-t-1), dtype=uint64)
            for m in range(m_max):
                bit = (mask & z[dim,m])>0
                print("%-2d"%bit,end='',flush=True)
            print("\n\t\t",end='',flush=True)
    print("\n\nznew",end='',flush=True)
    for j in range(d):
        dim = dvec[j]
        print('\n\tznew[dvec[%d]]\n\t\t'%j,end='',flush=True)
        for t in range(t_max):
            mask = array(1<<(t_max-t-1), dtype=uint64)
            for m in range(m_max):
                bit = (mask & znew[j,m])>0
                print("%-2d"%bit,end='',flush=True)
            print("\n\t\t",end='',flush=True)
    print("\nx (unrandomized)\n\t",end='',flush=True)
    for i in range(n):
        for j in range(d):
            print("%-7.3f"%x[i,j],end='',flush=True)
        print("\n\t",end='',flush=True)
    print("\nx (randomized)\n\t",end='',flush=True)
    for i in range(n):
        for j in range(d):
            print("%-7.3f"%xr[i,j],end='',flush=True)
        print("\n\t",end='',flush=True)
    print()
