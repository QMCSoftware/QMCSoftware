""" Sobol sequence generator """

from ..third_party.magic_point_shop import digitalseq_b2g
from copy import copy
from os import path
import sys
import mpmath
import pandas as pd


def bitreverse(a, m=None):
    """
    Reverse bit string of an integer.

    Args:
        a (int): Integer input.
        m (int): Length of bit string.

    Returns:
        int: Integer that corresponds to reversed bit string of ``a``

    """
    # https://tinyurl.com/yybvsmqe
    bin_number = bin(a)
    if m is None:
        m = len(bin_number) - 2
    reverse_number = bin_number[-1:1:-1]
    reverse_number = reverse_number + (m - len(reverse_number)) * "0"
    a_rev = int(reverse_number, 2)
    return a_rev


class DigitalSeq(digitalseq_b2g):
    """A fast digital sequence point generator based on generating matrices.

    This sequence generator can take classical (m by m) generating matrices
    or higher-order (alpha m by m) generating matrices from interlaced
    digital nets, interlaced polynomial lattice rules or higher-order
    polynomial lattice rules.  This code is specific for base 2 digital nets.

    This implementation is based upon, but faster than, ``digitalseq_b2g`` from:

    Reference:
        D. Nuyens, `The Magic Point Shop of QMC point generators and generating
        vectors.` MATLAB and Python software, 2018. Available from
        https://people.cs.kuleuven.be/~dirk.nuyens/

    """

    def __init__(self, Cs, kstart=0, m=None, s=None, returnDeepCopy=True):
        """Construct a digital sequence point generator given a list of
        generating matrices.

        Each generating matrix in the list is represented by a list of
        integer representation of its columns with the least significant
        bit at the top row.
        E.g., the upper triangular matrix with all ones is represented as
        [ 1, 3, 7, 15, ... ]
        The number of columns (i.e., the length of the list above
        representation for a generator matrix) determines m and then the
        number of points one can generate is 2**m.
        Only the length of the first generating matrix Cs[0] is checked,
        the others need to be at least this length.

        The number of bits needed to represent all columns of all generating
        matrices is the precision "t" with which the points will be
        constructed.

        For a classical net t = m as we have square m by m generating matrices.
        For higher order (e.g., through interlacing) nets t = alpha m where
        alpha is the rate of convergence aimed for in approximating integrals
        (in the function space setting used to construct or analyse the
        generating matrices).

        Args:
            Cs: generating matrices as a list of lists, see description above;
                or, if Cs is a string it will be interpreted as a filename
                and the generating matrices will be load from this file
            kstart: the index of the point from which you want this instance to
                start, the first point is 0

        Example:
            Example usage with a simply unit matrix and the powers of the
            polynomial
            (X+1) over Z_2[X]. The first dimension is then the van der Corput
            sequence. The second matrix is the choice of the second dimension of
            the sobol' and Niederreiter sequences.

            ::

            >>> from __future__ import print_function
            >>> m = 5
            >>> C1 = [ 2**i for i in range(m) ]
                # van der Corput sequence = identity matrix
            >>> C2 = [ 1 for i in range(m) ]
               # here we build the 2nd matrix of the sobol' and Niederreiter seq
            >>> for i in range(1, m): C2[i] = (C2[i-1] << 1) ^ C2[i-1]
            >>> Cs = [ C1, C2 ]
            >>> seq = digitalseq_b2g(Cs)
            >>> from copy import copy
            >>> [ copy(seq.cur) for x in seq ]

            [[0, 0], [16, 16], [24, 8], [8, 24], [12, 12], [28, 28], [20, 4],
            [4, 20], [6, 10], [22, 26], [30, 2], [14, 18], [10, 6], [26, 22],
            [18, 14], [2, 30], [3, 15], [19, 31], [27, 7], [11, 23], [15, 3],
            [31, 19], [23, 11], [7, 27], [5, 5], [21, 21], [29, 13], [13, 29],
            [9, 9], [25, 25], [17, 1], [1, 17]]

            ::

            >>> for x in seq:
            >>>     for xj in x: print(xj, end=" ")
            >>>     print()

            0 0
            0.5 0.5
            0.75 0.25
            0.25 0.75
            0.375 0.375
            0.875 0.875
            0.625 0.125
            0.125 0.625
            0.1875 0.3125
            0.6875 0.8125
            0.9375 0.0625
            0.4375 0.5625
            0.3125 0.1875
            0.8125 0.6875
            0.5625 0.4375
            0.0625 0.9375
            0.09375 0.46875
            0.59375 0.96875
            0.84375 0.21875
            0.34375 0.71875
            0.46875 0.09375
            0.96875 0.59375
            0.71875 0.34375
            0.21875 0.84375
            0.15625 0.15625
            0.65625 0.65625
            0.90625 0.40625
            0.40625 0.90625
            0.28125 0.28125
            0.78125 0.78125
            0.53125 0.03125
            0.03125 0.53125

        These are the first 32 sobol' or Niederreiter points in 2D.

        Warning: please mind the deepcopy if you store the member variables
        in a list, you get returned a reference to the value. If you don't
        use deepcopy all your list items will refer to the last value (and
        thus all be the same).

        Using numpy you can load a Bs.col (or Cs.col) file from the qmc4pde
        construction scripts by using:

        ::

          numpy.loadtxt("Bs64.col", int) # mind the int: read as integers!

        ::

          f = open("Bs.col")
          Bs = [map(int, line.split()) for line in f] # arbitrary big integers here...

        The easiest way is however to just provide the filename as the Cs
        argument (which will use method 2 above).

        Standard the points are generated as double precision numbers, however
        if one replaces seq.recipd by a multi precision variable with value
        2**-seq.t then the points will be delivered as such a type.
        Alternatively, the unscaled values are available as seq.cur and the
        points can then be generated as rationals with denominator 2**seq.t.

        Note: The generating matrices Cs which are passed in are available as
        the Csr field of this object, but note that they have been bit
        reversed.

        """
        basestr = str  # basestr for python2, str for python3
        if isinstance(Cs, basestr):
            abs_file_path = path.join(path.dirname(__file__),
                                      "../third_party/magic_point_shop", Cs)
            # filename passed in
            Cs = pd.read_csv(abs_file_path, header=None, delimiter=" ",
                             nrows=s).values.tolist()
        elif hasattr(Cs, "read"):
            # assume z is a stream like sys.stdin
            f = Cs
            Cs = [list(map(int, line.split())) for line in f]
            f.close()
        # otherwise Cs should be a list of generating matrices
        self.kstart = kstart
        if m is None:
            self.m = len(Cs[0])
        else:
            self.m = m
        if s is None:
            self.s = len(Cs)
        else:
            self.s = s
        self.t = max([int(a).bit_length() for a in Cs[0]])
        self.alpha = self.t / self.m
        self.Csr = [[bitreverse(int(Csjc), self.t) for Csjc in Csj] for Csj
                    in Cs]
        self.n = 2 ** self.m
        self.recipd = 2 ** -self.t
        self.returnDeepCopy = returnDeepCopy
        self.reset()

    def calc_next(self):
        """Calculate the next sequence point and update the index counter."""
        self.k = self.k + 1
        if self.k == 0: return True
        p = (((self.k ^ (self.k - 1)) + 1) >> 1)
        ctz = len(bin(p)[2:]) - 1
        for j in range(self.s):
            self.cur[j] = self.cur[j] ^ self.Csr[j][ctz]
            self.x[j] = self.recipd * self.cur[j]
        if self.k >= self.n: return False
        return True

    def __next__(self):
        """Return the next point of the sequence or raise StopIteration."""
        if self.k < self.n - 1:
            self.calc_next()
            return copy(self.x) if self.returnDeepCopy else self.x
        else:
            raise StopIteration


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = sys.stdin
    seq = DigitalSeq(filepath)
    if seq.t > 53:
        mpmath.mp.prec = seq.t
        seq.recipd = mpmath.mpf(1) / 2 ** seq.t
    for x in seq:
        for xj in x:
            print(xj, end=" ")
        print()
