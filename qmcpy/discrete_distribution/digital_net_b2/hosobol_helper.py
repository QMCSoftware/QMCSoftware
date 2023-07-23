import numpy as np
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
    if isinstance(a, (list, np.ndarray)):
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
    if isinstance(a, (list, np.ndarray)):
        return [bin_zfill(aa) for aa in a]

    len_max = bin_len_max(a)
    y = "0b" + dec2bin(a, len_max)
    return y


def bin_len_max(a):
    """
    It returns the maximum number of bits needed to represent the largest number in the integer class of a.

    Args:
      a: a signed or unsigned integer in Python or Numpy.

    Returns:
      The maximum number of bits that can be used to represent the largest integer in the integer class of a.
    """

    type_dict = {
        (np.uint8, np.int8): 8,
        (np.uint16, np.int16, np.half, np.float16): 16,
        (int, np.uint32, np.int32, np.single): 32,
        (np.uint64, np.int64, float, np.double): 64
    }

    for types, len_max in type_dict.items():
        if isinstance(a, types):
            return len_max

    raise ValueError(f"Unknown type of a, {type(a)}")


def bitget(a, bit):
    """
    Returns the value of the bit at position bit in a.

    Args:
        a: a signed or unsigned integer.
        bit: an integer between 1 (least significant bit) and the number of bits in the integer class of a.
             Or bit can be  a list of integers, with each element being an integer between 1 and the number of bits
             in the integer class of a.

    Returns:
        an integer or a list of integers representing the value of the bit at position bit in a.

    Raises:
        TypeError:  If a is not an integer.
        ValueError: If bit is not between 1 and the number of bits in the integer class of a.

    Reference:
        https://www.mathworks.com/help/matlab/ref/bitget.html

    Examples:
    >>> bit = list(range(8, 0, -1))
    >>> bitget(np.int8(127), bit)
    [0, 1, 1, 1, 1, 1, 1, 1]

    >>> bitget(np.uint8(255), bit)
    [1, 1, 1, 1, 1, 1, 1, 1]

    >>> bitget(np.int8(-29), bit)
    [1, 1, 1, 0, 0, 0, 1, 1]

    >>> bitget([np.int8(127), np.uint8(255), np.int8(-29)], 8)
    [0, 1, 1]

    >>> bitget([np.int8(127), np.uint8(255), np.int8(-29)], [8, 1])
    [[0, 1, 1], [1, 1, 1]]
    """
    a = list1_to_int(a)
    bit = list1_to_int(bit)

    if isinstance(bit, (list, np.ndarray)):
        return [bitget(a, b) for b in bit]

    if isinstance(a, (list, np.ndarray)):
        return [bitget(aa, bit) for aa in a]

    a = float_to_int(a)

    if isinstance(bit, int):
        if isinstance(a, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            if (bit > 0) and (bit <= len(bin_zfill(a)[2:])):
                one = np.array([1], dtype=type(a))[0]  # cast 1 to type a
                bit_minus_one = type(a)(bit - 1)
                return (a >> bit_minus_one) & one
            else:
                raise ValueError("bit must be between 1 and the number of bits in the integer class of a.")

    elif isinstance(a, (list, np.ndarray)):
        print("Contact QMCPy team for implementation of this case.")
    else:
        raise TypeError("a must be an integer or a list of integers.")


def bitset(a, bit, v=1):
    """
    Sets bit position bit in a to 1 or 0.

    Args:
        a: a signed or unsigned integer or an array of integers.
           If a is a double array, then each element is treated as an unsigned 64-bit integer.
           If a is a single array, then each element is treated as an unsigned 32-bit integer.
           If a is a NumPy float16 or half array, then each element is treated as an unsigned 16-bit integer.
        bit: an integer between 1 (least significant bit) and the number of bits in the integer class of a.
             If a is a double array, then all elements must be non-negative integers less than or equal to
             intmax('uint64'), and bit must be between 1 and 64.
        v: zero values of v sets the bit to 0 (off), and non-zero values of v sets the bit to 1 (on).

    Returns:
        an integer or a list of integers representing a with bit position bit set to 1.

    Raises:
        TypeError: If a is not an integer or a list of integers.
        ValueError: If bit is not between 1 and the number of bits in the integer class.

    Reference:
        https://www.mathworks.com/help/matlab/ref/bitset.html

    Examples:
    >>> bitset(np.uint8(9), 5, 1)
    25

    >>> bitset(4, [4, 5, 6])
    [12, 20, 36]

    >>> bitset(np.int8(75), 5)
    91

    >>> bitset(np.uint64(0), 1, 1)
    1

    >>> bitset(0., 1, 1)
    1

    >>> bitset([0, 1], 2, 1)
    [2, 3]

    >>> bitset([0, 1], 2, [1, 1])
    [2, 3]

    >>> bitset([0., 2.0], 1, [1])
    [1, 3]

    >>> bitset([0., 2.0], 1, 1)
    [1, 3]

    >>> bitset([2.0], 1, 1)
    3

    >>> bitset([2.0], 1, [1])
    3
    """

    a = list1_to_int(a)
    v = list1_to_int(v)
    bit = list1_to_int(bit)

    if isinstance(bit, (list, np.ndarray)):
        return [bitset(a, b, v) for b in bit]

    if isinstance(a, (list, np.ndarray)):
        if isinstance(v, (list, np.ndarray)):
            if len(a) == len(v):
                return [bitset(aa, bit, vv) for aa, vv in zip(a, v)]
            else:
                raise ValueError(f"Inputs a and v must have the same size")
        else:
            return [bitset(aa, bit, v) for aa in a]

    if isinstance(v, (list, np.ndarray)):
        return [bitset(a, bit, vv) for vv in v]

    a = float_to_int(a)

    if isinstance(a, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        if (bit > 0) and (bit <= len(bin_zfill(a)[2:])):
            # bitwise or of a and v left-shifted by (bit -1) and cast to type of a
            v = np.array([v], dtype=type(a))[0]
            bit_minus_1 = np.array([bit - 1], dtype=type(a))[0]
            return a | (v << bit_minus_1)
        else:
            raise ValueError(f"bit must be between 1 and the number of bits in the integer class of a.")


def list1_to_int(a):
    """
    If the input is a list or numpy array with only one element, return that element

    Args:
      a: list or numpy array of numbers

    Returns:
      the first element of the list if length of the list is 1. Otherwise the list itself.
    """
    if isinstance(a, (list, np.ndarray)) and (len(a) == 1):
        a = a[0]
    return a


def float_to_int(a):
    """
    If the input is a floating point number, convert it to an integer of the same size.

    If a is double, then it is cast as an unsigned 64-bit integer.
    If a is single, then it is cast as an unsigned 32-bit integer.
    If a is of type NumPy float16 or half, then it is cast as an unsigned 16-bit integer.

    Args:
      a: floating point

    Returns:
      integer after conversion
    """
    if np.issubdtype(type(a), np.floating):
        if isinstance(a, (np.half, np.float16)):
            a = np.uint16(a)
        elif isinstance(a, np.single):
            a = np.uint32(a)
        elif isinstance(a, (float, np.double)):
            a = np.uint64(a)
    return a