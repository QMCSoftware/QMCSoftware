""" Math functions used across various algorithms """

from numpy import max, abs


def tolfun(abstol, reltol, theta, mu, toltype):
    """
    Generalized error tolerance function.
    
    Args: 
        abstol (float): absolute error tolertance
        reltol (float): relative error tolerance
        theta (float): parameter in 'theta' case
        mu (loat): true mean
        toltype (str): different options of tolerance function
    """
    if toltype == 'combine': # the linear combination of two tolerances
        # theta=0---relative error tolarance
        # theta=1---absolute error tolerance
        tol  = theta*abstol + (1-theta)*reltol*abs(mu)
    elif toltype == 'max': # the max case
        tol  = max(abstol,reltol*abs(mu))
    return tol