import numpy as np

# TODO move utility functions to util


def dec2bin(a, numbits=1):
    """
    Produces a binary representation of an integer a with at least numbits bits.

    Args:
        a: an integer
        numbits: an integer representing the minimum number of bits in the binary representation

    Returns:
        a string of '0' and '1' representing the binary representation of a with at least numbits bits

    Raises:
        ValueError: If a is not an integer

    Reference:
        https://www.mathworks.com/help/matlab/ref/dec2bin.html

    Examples:
    >>> dec2bin(23)
    '10111'

    >>> dec2bin(23, 8)
    '00010111'

    >>> dec2bin(23,1)
    '10111'

    >>> dec2bin(-1, 8)
    '11111111'

    >>> dec2bin(-16, 8)
    '11110000'

    >>> dec2bin(0)
    '0'

    >>> dec2bin(10.5)
    '1010'

    >>> dec2bin('1')
    Traceback (most recent call last):
    ...
    ValueError: Unknown type of a, <class 'str'>

    >>> dec2bin([1023, 122, 14])
    ['1111111111', '0001111010', '0000001110']

    >>> dec2bin([12, 20, 36])
    ['001100', '010100', '100100']

    >>> dec2bin([-1, -16], 8)
    ['11111111', '11110000']
    """
    if type(a) in [list, np.ndarray]:
        max_abs_a = max(np.abs(a))
        numbits = max(numbits, len(dec2bin(max_abs_a)))
        return [dec2bin(aa, numbits) for aa in a]

    if np.issubdtype(type(a), np.floating):
        a = int(np.floor(a))

    if not np.issubdtype(type(a), np.integer):
        raise ValueError(f"Unknown type of a, {type(a)}")

    if a >= 0:
        return format(a, '0{}b'.format(numbits))
    else:
        return format(2**numbits + a, '0{}b'.format(numbits))


def bin_zfill(a):
    """
    It takes an integer and returns a string that represents the binary representation of that integer.

    Args:
      a: The number to be converted to binary

    Returns:
      The binary representation of the number a.

    Examples:
    >>> bin_zfill(23)
    '0b00000000000000000000000000010111'

    >>> bin_zfill(np.uint8(23))
    '0b00010111'

    >>> bin_zfill(np.uint16(23))
    '0b0000000000010111'

    >>> bin_zfill(np.int8(-1))
    '0b11111111'

    >>> bin_zfill(np.int8(-16))
    '0b11110000'

    >>> bin_zfill(np.half(10.5))
    '0b0000000000001010'

    >>> bin_zfill('1')
    Traceback (most recent call last):
    ...
    ValueError: Unknown type of a, <class 'str'>

    >>> bin_zfill(np.array([1023, 122, 14], dtype=np.uint16))
    ['0b0000001111111111', '0b0000000001111010', '0b0000000000001110']

    >>> bin_zfill(np.array([12, 20, 36], dtype=np.int8))
    ['0b00001100', '0b00010100', '0b00100100']

    >>> bin_zfill(np.array([-1, -16], dtype=np.int8))
    ['0b11111111', '0b11110000']
    """
    if type(a) in [list, np.ndarray]:
        return [bin_zfill(aa) for aa in a]

    l = bin_len_max(a)
    y = "0b" + dec2bin(a, l)
    return y

def bin_len_max(a):
    """
    It returns the maximum number of bits needed to represent the largest number in the integer class of a.

    Args:
      a: a signed or unsigned integer in Python or Numpy.

    Returns:
      The maximum number of bits that can be used to represent the largest integer in the integer class of a.
    """

    if type(a) in [np.uint8, np.int8]:
        len_max = 8
    elif type(a) in [np.uint16, np.int16, np.half, np.float16]:
        len_max = 16
    elif type(a) in [int, np.uint32, np.int32, np.single]:
        len_max = 32
    elif type(a) in [np.uint64, np.int64, float, np.double]:
        len_max = 64
    else:
        raise ValueError(f"Unknown type of a, {type(a)}")
    return len_max


def is_signed(a):
    """
    If a is an unsigned integer, return False. If a is a signed integer,
    return True. Otherwise, raise an error if the type of integer is not known.

    Args:
      a: a signed or unsigned integer in Python or Numpy.

    Returns:
      The function is_signed() returns a boolean value.
    """
    if type(a) in [np.uint8, np.uint16, np.uint32, np.uint64]:
        return False
    elif type(a) in [int, np.int8, np.int16, np.int32, np.int64]:
        return True
    else:
        raise ValueError("Unknown type of a, {type(a)}")


def bitget(a, bit):
    """
    Returns the value of the bit at position bit in a.

    Args:
        a: a signed or unsigned integer.
        bit: an integer between 1 (least significant bit) and the number of bits in the integer class of a. Or a list of integers,
             with each element being an integer between 1 and the number of bits in the integer class of a.

    Returns:
        an integer or a list of integers representing the value of the bit at position bit in a.

    Raises:
        TypeError:  If a is not an integer.
        ValueError: If bit is not between 1 and the number of bits in the integer class of a.

    Reference:
        https://www.mathworks.com/help/matlab/ref/bitget.html

    Examples:
    >>> bit = list(range(8, 0, -1))
    >>> bitget(np.uint8(255), bit)
    [1, 1, 1, 1, 1, 1, 1, 1]

    >>> bitget(np.int8(127), bit)
    [0, 1, 1, 1, 1, 1, 1, 1]

    >>> bitget(np.uint8(127), bit)
    [1, 1, 1, 1, 1, 1, 1, 1]
    """
    if type(bit) == list:
        return [bitget(a, b) for b in bit]

    l = bin_len_max(a)
    if type(bit) == int:
        if type(a) in [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            if (bit > 0) and (bit <= len(bin(a)[2:])):
                return (a >> (bit - 1)) & 1
            elif bit < l:
                return 0
            elif bit == l:
                return 0 if is_signed(a) else 1  # TODO debug
            else:
                raise ValueError("bit must be between 1 and the number of bits in the integer class of a.")

    elif type(a) in [list, np.ndarray]:
        print("Contact QMCPy team for implementation of this case.")
    else:
        raise TypeError("a must be an integer or a list of integers.")


def bitset(a, bit, v=1):
    """
    Sets bit position bit in a to 1 or 0.

    Args:
        a: a signed or unsigned integer or an array of integers.
            If a is a double array, then MATLAB® treats A as an unsigned 64-bit integer. TODO
        bit: an integer between 1 (least significant bit) and the number of bits in the integer class of a. If a is a double array, then all elements
        must be non-negative integers less than or equal to intmax('uint64'), and bit must be between 1 and 64.
        v: Zero values of v sets the bit to 0 (off), and non-zero values of v sets the bit to 1 (on).

    Returns:
        an integer or a list of integers representing a with bit position bit set to 1.

    Raises:
        TypeError: If a is not an integer or a list of integers.
        ValueError: If bit is not between 1 and the number of bits in the integer class of a or bit is not between 1 and 64
            if a is a list of integers.

    Reference:
        https://www.mathworks.com/help/matlab/ref/bitset.html

    Examples:
    >>> bitset(np.uint8(9), 5, 1)
    25

    >>> bitset(4, [4, 5, 6])
    [12, 20, 36]

    >>> bitset(np.int8(75), 5)
    91
    """
    if type(bit) == list:
        return [bitset(a, b, v) for b in bit]

    if type(a) in [float, int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
        if (bit > 0) and (bit <= len(bin_zfill(a)[2:])):
            return a | (v << (bit - 1))
        else:
            raise ValueError(f"bit must be between 1 and the number of bits in the integer class of a.")
    elif type(a) in [list, np.ndarray]:
        print("Contact QMCPy team for implementation of this case.")
        """
        if bit > 0 and bit <= 64:
            if all(isinstance(i, int) and i >= 0 and i <= int(2**64-1) for i in a):
                return [i | (1 << (bit-1)) for i in a]
            else:
                raise ValueError("If a is a list, all elements must be non-negative integers less than or equal to intmax('uint64'), and bit must be between 1 and 64.")
        else:
            raise ValueError("bit must be between 1 and 64.")
        """
    else:
        raise TypeError("a must be an integer or a list of integers.")

def MyHOSobol(m, s, d=1):
    """
    Higher-order Sobol sequence.

    Args:
        m: 2^m number of points
        s: dimension of final point set
        d: interlacing factor. Defaults to 1

    Returns:
        A matrix of higher order Sobol points, where the number of columns equals the dimension and the
        number of rows equals the number of points.

    Examples:
    >>> MyHOSobol(0, 1)
    array([[0.]])

    >>> MyHOSobol(1, 1)
    array([[0. ],
           [0.5]])

    >>> MyHOSobol(2, 1)
    array([[0.  ],
           [0.5 ],
           [0.75],
           [0.25]])

    >>> MyHOSobol(3, 4)
    array([[0.   , 0.   , 0.   , 0.   ],
           [0.5  , 0.5  , 0.5  , 0.5  ],
           [0.75 , 0.25 , 0.25 , 0.25 ],
           [0.25 , 0.75 , 0.75 , 0.75 ],
           [0.375, 0.375, 0.625, 0.875],
           [0.875, 0.875, 0.125, 0.375],
           [0.625, 0.125, 0.875, 0.625],
           [0.125, 0.625, 0.375, 0.125]])

    """

    z = np.loadtxt('sobol.dat')
    z = z[:2 ** m, :s * d]
    if d > 1:

        N = pow(2, m)  # Number of points
        u = 52
        depth = int(np.floor(u/d))
        
        # Create binary representation of digits
        W = z * pow(2, np.int64(depth))
        Z = np.floor(np.transpose(W))
        Y = np.zeros((s, N))
        for j in range(s):
            for i in range(depth):
                for k in range(d):
                    a = Z[j*d+k, :]
                    bit = depth - i
                    v = bitget(a, bit)

                    a = Y[j, :]
                    bit = (depth*d+1) - (k+1) - i*d
                    Y[j, :] = bitset(a, bit, v)
        Y = Y * pow(2, -depth*d)

        X = np.transpose(Y)
    else:
        X = z

    return X

if __name__ == "__main__":

    import doctest
    doctest.testmod()