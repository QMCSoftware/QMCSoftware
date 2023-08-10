from scipy.stats import norm
from numpy import *
def confInt(fn, n, d, discrete_distribution, inflation, level = 0.99):
    """
    Args:
        fn: the function to be passed
        n: the number of samples
        d: the dimension for which the confidence interval will be calculated
        discrete_distribution: the Discrete Distribution Object used to generate the samples
        inflation: the inflation factor for a conservative estimate
        level: confidence level for the interval
    """
    x = discrete_distribution.gen_samples(n)
    print("x = " + str(x))
    y = fn(x[:,d])
    print("y = " + str(y))
    mean = y.mean()
    print("mean = " + str(mean))
    variance = y.var(ddof=1)
    print("variance = " + str(variance))
    margin_error = sqrt(inflation) * norm.ppf(level) * (sqrt(variance/n))
    print("margin of error = " + str(margin_error))
    conf_int_lbound = mean - margin_error
    print("lower bound = " + str(conf_int_lbound))
    conf_int_hbound = mean + margin_error
    print("upper bound = " + str(conf_int_hbound))
    return conf_int_lbound,conf_int_hbound
    



    