"""
>>> from AsianCallFun import AsianCallFun as AsianCallFun
>>> acf = AsianCallFun()
>>> acf.volatility
0.5
>>> acf.S0
30
>>> acf.K
25
>>> acf.T
1
>>> acf.dimension
2
>>> acf.distribType
'uniform'
>>> acf.domainType
'box'
>>> acf.nominalValue
0
>>> acf.domain
array([[0, 0],
       [1, 1]])

>>> import numpy as np
>>> acf.f(np.array([[1, 2], [3, 4]]), [1, 2])
array([-4.15915193e-03,  3.93948451e-12])
"""