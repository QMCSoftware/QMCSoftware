import time
import numpy as np 
import numexpr as ne

def timer(fun,a,b):
    t1r,t1p = time.perf_counter(),time.process_time()
    c = fun(a,b)
    tfr,tfp = (time.perf_counter()-t1r),(time.process_time()-t1p)
    print(fun)
    print('CPU Time:', tfp)
    print('Elapsed Time:', tfr)
    return tfr, tfp, c

def numpyFun(a,b):
    c = 3*a+2*b
    #print(c,type(c))
    return c

def numexprFun(a,b):
    c = ne.evaluate('3*a+2*b')
    #print(c,type(c))
    return c

if __name__ == "__main__":
    a = np.arange(100000000)
    b = np.arange(100000000, 200000000)
    print()
    numpyRuntime, numpyCPUtime, np_result = timer(numpyFun,a,b)
    print()
    numexprRuntime, numexprCPUtime, ne_result  = timer(numexprFun,a,b)