import numpy as np

def format_bin(A):
    """
    It takes an integer and returns a string that represents the binary representation of that integer

    Args:
      A: The number to be converted to binary

    Returns:
      The binary representation of the number a.
    """
    if type(A) in [np.uint8, np.int8]:
        format_string = "{:08b}"
    elif type(A) in [np.uint16, np.int16]:
        format_string = "{:016b}"
    elif type(A) in [int, np.uint32, np.int32]:
        format_string = "{:032b}"
    elif type(A) in [np.uint64, np.int64]:
        format_string = "{:064b}"
    else:
        raise ValueError("Unknown type of A, {type(A)}")
    return "0b"+format_string.format(A)

def bin_len_max(A):
    """
    It returns the maximum number of bits needed to represent the largest number in the integer class of A.

    Args:
      A: a signed or unsigned integer in Python or Numpy.

    Returns:
      The maximum number of bits that can be used to represent the largest integer in the integer class of A.
    """
    if type(A) in [np.uint8, np.int8]:
        len_max = 8
    elif type(A) in [np.uint16, np.int16]:
        len_max = 16
    elif type(A) in [int, np.uint32, np.int32]:
        len_max = 32
    elif type(A) in [np.uint64, np.int64]:
        len_max = 64
    else:
        raise ValueError("Unknown type of A, {type(A)}")
    return len_max


def is_signed(A):
    """
    If the type of A is an unsigned integer, return False. If the type of A is a signed integer,
    return True. Otherwise, raise an error if the type of integer is not known.

    Args:
      A: a signed or unsigned integer in Python or Numpy.

    Returns:
      The function is_signed() returns a boolean value.
    """
    if type(A) in [np.uint8, np.uint16, np.uint32, np.uint64]:
        return False
    elif type(A) in [int, np.int8, np.int16, np.int32, np.int64]:
        return True
    else:
        raise ValueError("Unknown type of A, {type(A)}")


def bitget(A, BIT):
    """
    Returns the value of the bit at position BIT in A.

    Args:
        A: a signed or unsigned integer.
        BIT: an integer between 1 (least significant bit) and the number of bits in the integer class of A. Or a list of integers,
             with each element being an integer between 1 and the number of bits in the integer class of A.

    Returns:
        an integer or a list of integers representing the value of the bit at position BIT in A.

    Raises:
        TypeError:  If A is not an integer.
        ValueError: If BIT is not between 1 and the number of bits in the integer class of A.


    Examples:
    >>> bit = list(range(8, 0, -1))
    >>> bitget(np.uint8(255), bit)
    [1, 1, 1, 1, 1, 1, 1, 1]

    >>> bitget(np.int8(127), bit)
    [0, 1, 1, 1, 1, 1, 1, 1]

    >>> bitget(np.uint8(127), bit)
    [1, 1, 1, 1, 1, 1, 1, 1]
    """
    if type(BIT) == list:
        return [bitget(A, b) for b in BIT]

    l = bin_len_max(A)
    if type(BIT) == int:
        if type(A) in [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            if (BIT > 0) and (BIT <= len(bin(A)[2:])):
                return (A >> (BIT - 1)) & 1
            elif BIT < l:
                return 0
            elif BIT == l:
                return 0 if is_signed(A) else 1  # TODO debug
            else:
                raise ValueError("BIT must be between 1 and the number of bits in the integer class of A.")

    elif type(A) == list:
        print("Contact QMCPy team for implementation of this case.")
    else:
        raise TypeError("A must be an integer or a list of integers.")


def bitset(A, BIT, V=1):
    """
    Sets bit position BIT in A to 1 or 0.

    Args:
        A: a signed or unsigned integer or an array of integers.
        BIT: an integer between 1 (least significant bit) and the number of bits in the integer class of A. If A is a double array, then all elements
        must be non-negative integers less than or equal to intmax('uint64'), and BIT must be between 1 and 64.
        V: Zero values of V sets the bit to 0 (off), and non-zero values of V sets the bit to 1 (on).

    Returns:
        an integer or a list of integers representing A with bit position BIT set to 1.

    Raises:
        TypeError: If A is not an integer or a list of integers.
        ValueError: If BIT is not between 1 and the number of bits in the integer class of A or BIT is not between 1 and 64
            if A is a list of integers.

    Examples:
    >>> bitset(np.uint8(9), 5, 1)
    25
    """
    if type(BIT) == list:
        for b in BIT:
            A = bitset(A, b, V)
            return A

    l = bin_len_max(A)
    if type(A) in [int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
        if (BIT > 0) and (BIT <= len(format_bin(A)[2:])):
            return A | (V << (BIT - 1))
        elif BIT < l:
            return 0
        elif BIT == l:
            return 0 if is_signed(A) else 1  # TODO debug
        else:
            raise ValueError(f"BIT must be between 1 and the number of bits {l} in the integer class of A.")
    elif type(A) == list:
        print("Contact QMCPy team for implementation of this case.")
        """
        if BIT > 0 and BIT <= 64:
            if all(isinstance(i, int) and i >= 0 and i <= int(2**64-1) for i in A):
                return [i | (1 << (BIT-1)) for i in A]
            else:
                raise ValueError("If A is a list, all elements must be non-negative integers less than or equal to intmax('uint64'), and BIT must be between 1 and 64.")
        else:
            raise ValueError("BIT must be between 1 and 64.")
        """
    else:
        raise TypeError("A must be an integer or a list of integers.")

def MyHOSobol(m, s, d=1):
    # Higher order Sobol sequence
    # Create a higher order Sobol sequence.
    # 2^m number of points
    # s dimension of final point set
    # d interlacing factor
    # X Output Sobol sequence
    z = np.loadtxt('sobol.dat')
    z = z[:2 ** m, :s * d]
    if d > 1:
        print("Please contact the QMCPy team for this use case")
        """
        N = pow(2, m) # Number of points;
        u = 52
        depth = np.floor(u/d)
        
        # Create binary representation of digits;
        
        W = z * pow(2, np.int64(depth))
        Z = np.floor(np.transpose(W))
        Y = np.zeros((s, N))
        for j in range(s):
            for i in range(depth):
                for k in range(d):
                    Y[j,:] = bitset(Y[j,:], (depth*d+1) - k - (i-1)*d, bitget(Z[(j-1)*d+k,:], (depth+1) - i))
        Y = Y * pow(2, -depth*d)

        X = np.transpose(Y) # X is matrix of higher order Sobol points,
        # where the number of columns equals the dimension
        # and the number of rows equals the number of points;
        """
    else:
        X = z

    return X

if __name__ == "__main__":
    import doctest
    doctest.testmod()