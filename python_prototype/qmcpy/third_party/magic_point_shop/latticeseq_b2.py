#!/usr/bin/env python
from __future__ import print_function

####
## (C) Dirk Nuyens, KU Leuven, 2016,2017,...
##

# the following function is duplicated from poylat.py such that this file can be used stand alone
def bitreverse(a, m=None):
    a_bin = "{0:b}".format(a)
    a_m = len(a_bin)
    if m == None: m = a_m
    a_rev = int(a_bin[::-1], 2) << max(0, m - a_m)
    return a_rev

# generating vector from
#   Constructing embedded lattice rules for multivariate integration
#   R Cools, FY Kuo, D Nuyens -  SIAM J. Sci. Comput., 28(6), 2162-2188.
# maximum number of points was set to 2**20, maximum number of dimensions is 250
# constructed for unanchored Sobolev space with order dependent weights of order 2,
# meaning that all 2-dimensional projections are taken into account explicitly
# (in this case all choices of weights are equivalent and this is thus a generic
# order 2 rule)
exod2_base2_m20_CKN_z = [1, 182667, 469891, 498753, 110745, 446247, 250185, 118627, 245333, 283199, \
        408519, 391023, 246327, 126539, 399185, 461527, 300343, 69681, 516695, 436179, 106383, 238523, \
        413283, 70841, 47719, 300129, 113029, 123925, 410745, 211325, 17489, 511893, 40767, 186077, \
        519471, 255369, 101819, 243573, 66189, 152143, 503455, 113217, 132603, 463967, 297717, 157383, \
        224015, 502917, 36237, 94049, 170665, 79397, 123963, 223451, 323871, 303633, 98567, 318855, \
        494245, 477137, 177975, 64483, 26695, 88779, 94497, 239429, 381007, 110205, 339157, 73397, \
        407559, 181791, 442675, 301397, 32569, 147737, 189949, 138655, 350241, 63371, 511925, 515861, \
        434045, 383435, 249187, 492723, 479195, 84589, 99703, 239831, 269423, 182241, 61063, 130789, \
        143095, 471209, 139019, 172565, 487045, 304803, 45669, 380427, 19547, 425593, 337729, 237863, \
        428453, 291699, 238587, 110653, 196113, 465711, 141583, 224183, 266671, 169063, 317617, 68143, \
        291637, 263355, 427191, 200211, 365773, 254701, 368663, 248047, 209221, 279201, 323179, 80217, \
        122791, 316633, 118515, 14253, 129509, 410941, 402601, 511437, 10469, 366469, 463959, 442841, \
        54641, 44167, 19703, 209585, 69037, 33317, 433373, 55879, 245295, 10905, 468881, 128617, 417919, \
        45067, 442243, 359529, 51109, 290275, 168691, 212061, 217775, 405485, 313395, 256763, 152537, 326437, \
        332981, 406755, 423147, 412621, 362019, 279679, 169189, 107405, 251851, 5413, 316095, 247945, 422489, \
        2555, 282267, 121027, 369319, 204587, 445191, 337315, 322505, 388411, 102961, 506099, 399801, 254381, \
        452545, 309001, 147013, 507865, 32283, 320511, 264647, 417965, 227069, 341461, 466581, 386241, \
        494585, 201479, 151243, 481337, 68195, 75401, 58359, 448107, 459499, 9873, 365117, 350845, 181873, \
        7917, 436695, 43899, 348367, 423927, 437399, 385089, 21693, 268793, 49257, 250211, 125071, 341631, \
        310163, 94631, 108795, 21175, 142847, 383599, 71105, 65989, 446433, 177457, 107311, 295679, 442763, \
        40729, 322721, 420175, 430359, 480757]

class latticeseq_b2:

    def __init__(self, z=exod2_base2_m20_CKN_z, kstart=0, m=None, s=None, returnDeepCopy=True):
        import sys
        basestr = str # basestr for python2, str for python3
        if isinstance(z, basestr):
            # filename passed in
            f = open(z)
            z = [ int(line) for line in f ]
        elif hasattr(z, 'read'):
            # assume z is a stream like sys.stdin
            f = z
            z = [ int(line) for line in f ]
        # otherwise z should be a list of int's or something equivalent
        self.kstart = kstart
        if m == None: self.m = 32
        else: self.m = m
        if s == None: self.s = len(z)
        else: self.s = s
        self.z = z[:self.s]
        self.n = 2**self.m
        self.scale = 2**-self.m
        self.returnDeepCopy = returnDeepCopy
        self.x = [ 0 for j in range(self.s) ]
        self.reset()

    def reset(self):
        """Reset this lattice sequence to its initial state: next index = kstart."""
        self.set_state(self.kstart)

    def set_state(self, k):
        """Set the index of the next point to k."""
        self.k = k - 1
        self.calc_next()
        self.k = k - 1

    def calc_next(self):
        """Calculate the next sequence point and update the index counter."""
        from math import floor
        self.k = self.k + 1
        phik = bitreverse(self.k, self.m) * self.scale
        for j in range(self.s):
            self.x[j] = phik * self.z[j]
            self.x[j] = self.x[j] - floor(self.x[j])
        if self.k >= self.n: return False
        return True

    def calc_block(self, m):
        """Calculate all points from 2**(m-1) up to 2**m, without using radical
        inverse, using numpy and returning as numpy array object. This is tons
        faster!"""
        from numpy import arange, outer
        n = 2**m
        start = min(1, n/2) # this is a funky way of setting start to zero for m == 0
        # the arange below only ranges over odd numbers, except for m == 0, then we only have 0
        x = (outer(arange(min(1,n/2), n, 2, dtype='i'), self.z) % n) / float(n)
        return x

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        """Return the next point of the sequence or raise StopIteration."""
        if self.k < self.n - 1:
            self.calc_next()
            if self.returnDeepCopy:
                from copy import deepcopy
                return deepcopy(self.x)
            return self.x
        else:
            raise StopIteration

    def next(self):
        return self.__next__()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2: raise ValueError("Please specify the power of 2 as a command line argument: latticeseq_b2 m < z.txt")
    try:
        m = int(sys.argv[1])
    except:
        raise ValueError("Please specify the power of 2 as a command line argument: latticeseq_b2 m < z.txt")
    if len(sys.argv) > 2: f = sys.argv[2]
    else: f = sys.stdin
    seq = latticeseq_b2(f, m=m)
    for x in seq:
        for xj in x:
            print(xj, end=" ")
        print()
