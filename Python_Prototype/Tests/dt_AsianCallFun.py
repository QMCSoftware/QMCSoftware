"""
>>> from AsianCallFun import AsianCallFun as AsianCallFun
>>> acf = AsianCallFun() # doctest:+ELLIPSIS
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
>>> import numpy as np; acf.f(np.array([[1, 2], [3, 4]]), [1, 2])
array([-4.15915193e-03,  3.93948451e-12])

"""