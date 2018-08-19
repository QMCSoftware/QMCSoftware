"""
>>> from Keister import Keister as Keister; kf = Keister(); # doctest:+ELLIPSIS
>>> kf.dimension
2
>>> kf.distribType
'uniform'
>>> kf.domainType
'box'
>>> kf.nominalValue
0
>>> kf.domain
array([[0, 0],
       [1, 1]])
>>> import numpy as np; kf.f(np.array([[1, 2], [3, 4]]), [1, 2])
array([-4.15915193e-03,  3.93948451e-12])

"""
