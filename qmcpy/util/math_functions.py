""" Math functions used across various algorithms. """

def _tol_fun(abs_tol, rel_tol, theta, mu, toltype):
    """
    Generalized error tolerance function.

    Args:
        abs_tol (float): absolute error tolerance
        rel_tol (float): relative error tolerance
        theta (float): parameter in 'theta' case
        mu (float): true mean
        toltype (str): different options of tolerance function

    Return:
        float: tolerance as weighted sum of absolute and relative tolerance
    """
    if toltype == 'combine':  # the linear combination of two tolerances
        # theta == 0 --> relative error tolerance
        # theta === 1 --> absolute error tolerance
        tol = theta * abs_tol + (1 - theta) * rel_tol * abs(mu)
    elif toltype == 'max':  # the max case
        tol = max(abs_tol, rel_tol * abs(mu))
    return tol
