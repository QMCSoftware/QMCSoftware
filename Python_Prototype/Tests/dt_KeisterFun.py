"""
>>> from KeisterFun import KeisterFun as KeisterFun
>>> kf = KeisterFun()
>>> kf.dimension
2
>>> kf.domainType
'box'
>>> kf.distrib['name']
'IIDZGaussian'
>>> kf.nominalValue
0
>>> import numpy as np
>>> kf.g(np.array([[1, 2], [3, 4]]), [1, 2])
array([-1.93921993,  0.89115104]) 
"""
