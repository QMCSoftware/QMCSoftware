from qmcpy import *
a,b = [0.,-1.],[2.,1.]
dd = Sobol(dimension=2,seed=7) # discrete distribution
m = Uniform(dd,lower_bound=a,upper_bound=b) # true measure
i = CustomFun(m,lambda x: x[:,0]*x[:,1]**2) # integrand
sc = CubQmcSobolG(i,abs_tol=1e-4) # stopping criterion
solution,data = sc.integrate()
print(solution)
print(data)