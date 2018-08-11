import importlib



from KeisterFun import KeisterFun
import numpy as np
kf = KeisterFun()
kf.f(np.array([[1, 2], [3, 4]]), [1, 2])
# array([ -4.15915193e-03,   3.93948451e-12])

import accumData
accumData = importlib.reload(accumData)
import meanVarData
meanVarData = importlib.reload(meanVarData)
from meanVarData import meanVarData
mvd = meanVarData()
print(mvd.__dict__)
print(mvd.timeStart)


#from CLTStopping import CLTStopping
#c = CLTStopping()