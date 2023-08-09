from scipy.stats import norm
from numpy import *
# using function x^2 + 8 
# x is a uniform random variable
def confInt(x, inflation,level = 0.95):
    x = atleast_1d(x)
    sum = 0
    for i in range(x.size):
        sum = sum + ((x[i])** 2) + 8
    mean = sum / x.size
    print("mean = " + str(mean))
    var_sum = 0
    for i in range(x.size):
        var_sum = var_sum + ((((x[i])** 2) + 8) - mean) ** 2
    variance = var_sum / (x.size - 1)
    print("variance = " + str(variance))
    margin_error = sqrt(inflation) * norm.ppf(level) * (sqrt(variance/x.size))
    print("margin of error = " + str(margin_error))
    conf_int_lbound = mean - margin_error
    print("lower bound = " + str(conf_int_lbound))
    conf_int_hbound = mean + margin_error
    print("upper bound = " + str(conf_int_hbound))
    return conf_int_lbound,conf_int_hbound
    



    