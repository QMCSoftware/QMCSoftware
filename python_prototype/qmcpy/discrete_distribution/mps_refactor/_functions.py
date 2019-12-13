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