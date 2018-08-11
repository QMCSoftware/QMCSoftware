"""
dt_qmc
doctest for QMC Community Software

$ python dt_qmc.py -v
"""

import KeisterFun
import meanVarData
import dt_CLTStopping

if __name__ == '__main__':
    import doctest

    doctest.testmod(KeisterFun)
    doctest.testmod(meanVarData)
    #doctest.testmod(dt_CLTStopping)

