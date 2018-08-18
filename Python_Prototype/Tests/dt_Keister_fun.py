>>> import numpy as np;from KeisterFun import KeisterFun;kf = KeisterFun();print(kf.__dict__)
{'domain': array([[0, 0],
           [1, 1]]), 'domainType': 'box', 'dimension': 2, 'distribType': 'uniform', 'nominalValue': 0}
>>> print(kf.dimension)
2
>>> kf.f(np.array([[1, 2], [3, 4]]), [1, 2])
array([-4.15915193e-03,  3.93948451e-12])
