from numpy import *
import os

def sobol_ags(n, d, randomize='LMS', skip=0, gc=False, f='gen_mat.21201.32.lsb.npy'):
    """ 
    Sobol' generator
    
    Args:
        n (int): number of samples
        d (int): dimension of sampels
        skip (int): number of samples to skip
        randomize (str): one of ['None', 'Digital Shift', 'Linear Matrix Scramble', 'Nested Uniform Scramble']
        f (str): file name with convention (1).(2).(3).(4).npy such that 
            - (1) any name
            - (2) max dimension
            - (3) m such that max samples = 2^m
            - (4) 'lsb' for least significant bit in 0th column of j^th dimension generating vector or 
                  'msb' for most significant ... 
                  Note: 'msb' is faster
            Default generator from: https://web.maths.unsw.edu.au/~fkuo/sobol/
    """
    root = os.path.dirname(__file__)
    z = load(root+'/generating_matricies/%s'%f).astype(uint32)
    f_split = f.split('.')
    d_max = int(f_split[1])
    m_max = int(f_split[2])
    sb = f_split[3] # most significant bit (msb) or least significant bit (lsb) in column 0 of C_j
    scale = 2**(-m_max)
    if (z.shape!=(d_max,m_max)) or (sb not in ['lsb','msb']):
        raise Exception("generating matrix file name does not follow convention.")
    # initialize randomization
    z = z[:d,:] # only need the first n rows
    randomize = randomize.upper() if type(randomize)==str else 'NONE'
    if randomize in ['LMS','LINEAR MATRIX SCRAMBLE']:
        scramble_mat = zeros((m_max,m_max),dtype=uint8)
        z_mat = zeros((m_max,m_max),dtype=uint8)
        pows2 = 2**(arange(m_max-1,-1,-1,dtype=uint32) if sb=='msb' else arange(m_max,dtype=uint32)).reshape((-1,1))
        for j in range(d):
            scramble_mat[:,:] = 0
            z_mat[:,:] = 0
            # set lower triangular scrambling matrix
            for k1 in range(m_max):
                for k2 in range(k1):
                    scramble_mat[k1,k2] = random.randint(2)
            scramble_mat += diag(ones(m_max,dtype=uint8))
            # expand C_j along each colum with more significant bits higher in matrix
            for c in range(m_max):
                for r in range(m_max):
                    s = int(m_max-r-1) if sb=='msb' else int(r)
                    z_mat[r,c] = (z[j,c]>>s)&1
            # left multiply scrambling matrix by generating vectors
            z_mat = dot(scramble_mat,z_mat)%2
            z[j] = (z_mat * pows2).sum(0)
        randomize = 'DS' # also must apply digital shift
    if randomize in ['DS','DIGITAL SHIFT']:
        rshift = random.randint(2**m_max,size=d)
    if randomize in ['NUS','NESTED UNIFORM SCRAMBLE']:
        raise Exception("Nested Uniform Scramble not yet implemented.")
    # generate points
    x = zeros((n,d),dtype=double)
    for i in range(skip,n+skip):
        m = 0
        maskv = zeros(d,dtype=uint32)
        im = i^(i>>1) if gc else i
        while im!=0 and m<m_max:
            if im&1:
                maskv = maskv ^ z[:,m]
            im = im>>1
            m += 1
        if sb=='lsb':
            # flip bits
            maskv_r = zeros(d,dtype=uint32) # store maskv bit-reversed
            for j in range(d):
                for b in range(m_max):
                    maskv_r[j] += ((maskv[j]>>b)&1) * (2**(m_max-1-b))
            maskv = maskv_r
        # randomize this point
        if randomize in ['DS','DIGITAL SHIFT']:
            maskv = maskv ^ rshift
        # scale this point
        x[i-skip,:] = maskv*scale
    return x

def lsb_to_msb(f):
    """ Convert a generating matrix with LSB to MSB. """
    root = os.path.dirname(__file__)
    fp = root+'/generating_matricies/%s'%f
    z = load(fp).astype(uint32)
    z2 = zeros(z.shape,dtype=uint32)
    d,m = z.shape
    for r in range(d):
        if r%1000==0: print('Completed %d'%r)
        for c in range(m):
            for b in range(m):
                z2[r,c] += ((z[r,c]>>b)&1) * (2**(m-1-b))
    save(fp.replace('lsb','msb'),z2)


if __name__ == '__main__':
    # params
    n = 2**6
    d = 2
    randomize = 'LMS'
    skip = 0
    gc = False
    f = 'gen_mat.21201.32.msb.npy' # ['gen_mat.21201.32.msb.npy', 'gen_mat.21201.32.lsb.npy', 'gen_mat.51.30.msb.npy']
    x_cuts = 8
    y_cuts = 8
    # get sobol points
    x = sobol_ags(n, d, randomize, skip, gc, f) 
    print(x)
    # plot
    from matplotlib import pyplot
    fig,ax = pyplot.subplots(figsize=(5,5))
    ax.scatter(x[:,0],x[:,1],color='b',s=10)
    for ix in arange(1,x_cuts,dtype=float): ax.axvline(x=ix/x_cuts,color='r')
    for iy in arange(1,y_cuts,dtype=float): ax.axhline(y=iy/y_cuts,color='r')
    ax.set_xlim([0,1])
    ax.set_xticks([0,1])
    ax.set_ylim([0,1])
    ax.set_yticks([0,1])
    ax.set_aspect(1)
    pyplot.savefig('_ags/temp.png') 
    